"""Chrome DevTools MCP 的只读检索适配层；不向模型开放任意浏览器操作。"""
from __future__ import annotations

import asyncio
import base64
import fcntl
import ipaddress
import json
import os
import re
import socket
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from app.services.privacy import PrivacySanitizer
from app.services.web_access import WebResearchResult, WebSource, public_domain, search_policy


class BrowserAccessError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@contextmanager
def browser_lease():
    directory = Path.home() / ".cache/caretrace"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "browser-research.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BrowserAccessError("busy") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def browser_error_code(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        codes = [browser_error_code(item) for item in error.exceptions]
        return next((code for code in codes if code != "unavailable"), "unavailable")
    if isinstance(error, BrowserAccessError):
        return error.code
    return "timeout" if isinstance(error, TimeoutError) else "unavailable"


def safe_public_url(url: str, domain: str = "") -> str:
    """直接导航前验证目标，不允许内网、登录、账号管理、私信和动作链接。"""
    try:
        host = public_domain(url)
        parts = urlsplit(url)
    except ValueError as exc:
        raise BrowserAccessError("blocked") from exc
    if domain and host != domain and not host.endswith("." + domain):
        raise BrowserAccessError("blocked")
    path = unquote(parts.path).lower()
    if re.search(r"(?:^|/)(?:login|signin|logout|signout|account|settings|messages?|inbox|chat|user|profile|oauth|admin|delete|follow|like|publish)(?:/|$)", path):
        raise BrowserAccessError("blocked")
    if any(key.lower() in {"redirect", "redirect_uri", "url", "next", "callback", "returnurl"} for key in parse_qs(parts.query)):
        raise BrowserAccessError("blocked")
    try:
        addresses = socket.getaddrinfo(parts.hostname, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise BrowserAccessError("unavailable") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise BrowserAccessError("blocked")
    return url


def result_url(url: str) -> str:
    """Bing 的公开跳转链接只在本地解码，随后仍须通过导航校验。"""
    parsed = urlsplit(url)
    if (parsed.hostname or "").endswith(".bing.com") and parsed.path == "/ck/a":
        value = parse_qs(parsed.query).get("u", [""])[0]
        if value.startswith("a1"):
            try:
                encoded = value[2:]
                return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
            except (ValueError, UnicodeError):
                return ""
    return url


def _result_text(result) -> str:
    if getattr(result, "isError", False):
        raise BrowserAccessError("unavailable")
    return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")


def _json_result(text: str) -> dict:
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.S)
    try:
        value = json.loads(match[1] if match else text)
        if isinstance(value, dict):
            return value
    except (ValueError, TypeError):
        pass
    raise BrowserAccessError("unavailable")


def local_chrome_endpoint() -> str:
    """兼容 Chrome 的内置远程调试：部分版本仅提供 WS，不开放 /json/version。"""
    for path in (Path.home() / "Library/Application Support/Google/Chrome/DevToolsActivePort",
                 Path.home() / ".config/google-chrome/DevToolsActivePort"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            port = int(lines[0])
            if 1 <= port <= 65535 and re.fullmatch(r"/devtools/browser/[a-zA-Z0-9-]+", lines[1]):
                return f"ws://127.0.0.1:{port}{lines[1]}"
        except (OSError, ValueError, IndexError):
            continue
    return ""


def extraction_script(host: str, *, detail: bool, paths: list[str] | None = None) -> str:
    # host 来自已校验目标；不插入网页文本、用户内容或模型生成的 JavaScript。
    return ("() => { const expectedHost = " + json.dumps(host) + "; const detail = " + json.dumps(detail)
            + "; const expectedPaths = " + json.dumps(paths or []) + ";") + r"""
      const host = location.hostname.replace(/^www\./, '');
      if (location.protocol !== 'https:' || location.port && location.port !== '443') return {access: 'blocked'};
      if (host !== expectedHost && !(expectedHost === 'bing.com' && host === 'cn.bing.com')) return {access: 'blocked'};
      const path = decodeURIComponent(location.pathname);
      if (/\/(login|signin|account|settings|messages?|inbox|user|profile)(\/|$)/i.test(path))
        return {access: 'login_required'};
      if (expectedPaths.length && !expectedPaths.includes(path)) return {access: 'blocked'};
      const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      const clean = (value, limit) => String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit);
      const dialog = Array.from(document.querySelectorAll('[role="dialog"], .login-container, .login-modal, .captcha, #captcha')).filter(visible);
      if (dialog.some(el => /验证码|安全验证|拖动滑块|captcha|verify you are/i.test(el.innerText))) return {access: 'challenge'};
      if (dialog.some(el => /登录|login|sign in/i.test(el.innerText))) return {access: 'login_required'};
      const root = document.querySelector('main, article, #content, #b_results, .feeds-page, .search-result-container');
      const pageText = clean(root?.innerText, 3500);
      if (/安全验证|完成验证后|verify you are human|验证码验证/.test(document.title + pageText.slice(0, 150))) return {access: 'challenge'};
      let links = [];
      if (host.endsWith('xiaohongshu.com')) {
        links = Array.from(document.querySelectorAll('.note-item')).map(el => {
          const candidates = Array.from(el.querySelectorAll('a[href*="/explore/"], a[href*="/search_result/"]'));
          // 搜索卡片可能同时有无 token 的封面链接和带公开访问 token 的正文链接。
          const a = candidates.find(link => new URL(link.href).searchParams.has('xsec_token')) || candidates[0];
          return {title: clean(el.querySelector('.title, .note-title')?.innerText, 100),
                  url: a?.href || '', excerpt: clean(el.querySelector('.title, .note-title')?.innerText, 300)};
        });
      } else if (host.endsWith('bing.com')) {
        links = Array.from(document.querySelectorAll('#b_results .b_algo')).map(el => {
          const a = el.querySelector('h2 a');
          return {title: clean(a?.innerText, 100), url: a?.href || '', excerpt: clean(el.querySelector('.b_caption p')?.innerText, 400)};
        });
      } else {
        links = Array.from((root || document.createElement('main')).querySelectorAll('a[href]')).filter(visible).map(a => ({
          title: clean(a.innerText, 100), url: a.href, excerpt: clean(a.closest('article, .SearchResult-Card, .ContentItem')?.innerText, 400)
        }));
      }
      const body = host.endsWith('xiaohongshu.com')
        ? document.querySelector('#detail-desc, .note-text')
        : document.querySelector('article, .RichContent-inner, .topic-content, #js_content, main');
      const title = document.querySelector('#detail-title, h1')?.innerText || document.title;
      return {access: 'ok', url: location.href, title: clean(title, 100),
              text: detail ? clean(body?.innerText, 2200) : '',
              links: links.filter(x => x.url && x.title.length > 3).slice(0, 20)};
    }"""


class ChromeResearchClient:
    def __init__(self, settings):
        self.settings = settings
        self.calls: list[dict] = []
        self.page_id: int | None = None
        self.reset_required = False

    def server_parameters(self):
        from mcp import StdioServerParameters

        args = [*self.settings.web_access_mcp_args, "--no-usage-statistics", "--no-performance-crux",
                "--no-category-performance", "--no-category-network", "--no-category-emulation",
                "--page-id-routing", "--no-file-navigations", "--no-source-maps",
                "--no-category-input", "--no-category-memory"]
        endpoint = self.settings.web_access_browser_url or local_chrome_endpoint()
        if endpoint:
            parsed = urlsplit(endpoint)
            valid_path = (parsed.path in {"", "/"} if parsed.scheme == "http"
                          else bool(re.fullmatch(r"/devtools/browser/[a-zA-Z0-9-]+", parsed.path)))
            if (parsed.scheme not in {"http", "ws"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                    or parsed.username or parsed.password or not valid_path or parsed.query or parsed.fragment):
                raise BrowserAccessError("blocked")
            args += ["--browserUrl" if parsed.scheme == "http" else "--wsEndpoint", endpoint]
        else:
            args += ["--autoConnect"]
        # 不把数据库、LLM、SMTP 密钥传给第三方浏览器工具进程。
        env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "TMPDIR", "LOCALAPPDATA", "SYSTEMROOT") if key in os.environ}
        if self.settings.web_access_node_bin:
            env["PATH"] = self.settings.web_access_node_bin + os.pathsep + env.get("PATH", "")
        env.update({"npm_config_registry": "https://registry.npmjs.org", "CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1",
                    "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1"})
        return StdioServerParameters(command=self.settings.web_access_mcp_command, args=args, env=env)

    @asynccontextmanager
    async def _session(self):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        # 同一任务拥有完整的 AnyIO cancel scope 生命周期。
        with open(os.devnull, "w") as errlog:
            async with stdio_client(self.server_parameters(), errlog=errlog) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=max(65, self.settings.web_access_tool_timeout_seconds))) as session:
                    await session.initialize()
                    available = {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}
                    if any("pageId" not in available.get(name, {}).get("properties", {})
                           for name in ("navigate_page", "evaluate_script", "close_page")):
                        raise BrowserAccessError("unavailable")
                    yield session

    async def _call(self, session, name: str, arguments: dict) -> str:
        if name not in {"new_page", "navigate_page", "evaluate_script", "close_page"}:
            raise BrowserAccessError("blocked")
        started = time.perf_counter()
        record = {"tool": name, "status": "failed"}
        try:
            timeout = max(60, self.settings.web_access_tool_timeout_seconds) if name == "new_page" else self.settings.web_access_tool_timeout_seconds
            if name == "close_page" and self.reset_required:
                timeout = min(3, timeout)
            async with asyncio.timeout(timeout):
                result = _result_text(await session.call_tool(name, arguments=arguments))
            record["status"] = "completed"
            return result
        except TimeoutError as exc:
            self.reset_required = True
            if name == "new_page":
                raise BrowserAccessError("unavailable") from exc
            raise
        finally:
            # new_page/close_page 会返回浏览器标签列表，禁止把原始返回写入 trace。
            record["durationMs"] = round((time.perf_counter() - started) * 1000, 1)
            self.calls.append(record)

    async def _navigate(self, session, url: str, domain: str = "") -> None:
        await asyncio.to_thread(safe_public_url, url, domain)
        # 1.x 已移除旧版 allowList 参数。只发送已验证目标，并在 DOM 读取前
        # 核对重定向后的域名和路径；这不是浏览器网络层的重定向拦截器。
        await self._call(session, "navigate_page", {"pageId": self.page_id, "type": "url", "url": url,
                         # MCP waits for navigation twice (goto + event waiter), then
                         # settles the DOM. Leave room inside the transport deadline.
                         "timeout": max(500, min(10000, int((self.settings.web_access_tool_timeout_seconds - 5) * 500))),
                         "handleBeforeUnload": "dismiss"})

    async def _extract(self, session, url: str, *, detail: bool) -> dict:
        host = urlsplit(url).hostname.removeprefix("www.")
        paths = [unquote(urlsplit(url).path)]
        if detail and host == "xiaohongshu.com":
            note = re.fullmatch(r"/(?:explore|search_result)/([a-zA-Z0-9]+)", paths[0])
            if note:
                paths = [f"/{prefix}/{note[1]}" for prefix in ("explore", "search_result")]
        data = {}
        # 动态页面只做有界等待，不反复滚动或截图。
        for attempt in range(4):
            data = _json_result(await self._call(session, "evaluate_script", {
                "pageId": self.page_id, "function": extraction_script(host, detail=detail, paths=paths),
                "dialogAction": "dismiss", "waitForStableDom": False}))
            if data.get("access") != "ok":
                raise BrowserAccessError(data.get("access", "blocked"))
            if data.get("text" if detail else "links"):
                break
            if attempt < 3:
                await asyncio.sleep(1)
        return data

    async def search(self, query: str, site: str) -> WebResearchResult:
        try:
            with browser_lease():
                async with asyncio.timeout(self.settings.web_access_timeout_seconds):
                    async with self._session() as session:
                        return await self.research_in_session(session, query, site)
        except BaseExceptionGroup as exc:
            raise BrowserAccessError(browser_error_code(exc)) from exc

    async def research_in_session(self, session, query: str, site: str) -> WebResearchResult:
        self.calls = []
        self.page_id = None
        self.reset_required = False
        try:
            created = await self._call(session, "new_page", {"url": "about:blank", "background": True})
            selected = re.search(r"^\s*(\d+):\s+about:blank\s+\[selected\]", created, re.M)
            if not selected:
                raise BrowserAccessError("unavailable")
            self.page_id = int(selected[1])
            return await self._research(session, query, site)
        finally:
            if self.page_id is not None:
                try:
                    await self._call(session, "close_page", {"pageId": self.page_id})
                except Exception:
                    pass

    async def _research(self, session, query: str, site: str) -> WebResearchResult:
        spec = search_policy()["sites"].get(site, {})
        search_url = (spec["search"].format(query=quote(query)) if "search" in spec
                      else "https://www.bing.com/search?q=" + quote((f"site:{site} " if site else "") + query))
        await self._navigate(session, search_url)
        listing = await self._extract(session, search_url, detail=False)
        sources = []
        seen = set()
        privacy = PrivacySanitizer()
        for item in listing.get("links", [])[:20]:
            url = result_url(str(item.get("url", "")))
            try:
                await asyncio.to_thread(safe_public_url, url, site)
            except BrowserAccessError:
                continue
            parts = urlsplit(url)
            # 小红书只读取笔记，避免关注个人主页或搜索导航；知乎只读取问答/文章。
            if site == "xiaohongshu.com" and not re.match(r"/(?:explore|search_result)/[a-zA-Z0-9]+", parts.path):
                continue
            if site == "zhihu.com" and not re.match(r"/(?:question/\d+|p/\d+)", parts.path):
                continue
            key = (parts.hostname, parts.path)
            title = privacy.sanitize(str(item.get("title", "")))[:100].strip()
            if key in seen or not title:
                continue
            seen.add(key)
            sources.append(WebSource(title, url, privacy.sanitize(str(item.get("excerpt", "")))[:400]))
            if len(sources) >= self.settings.web_access_max_results:
                break
        for source in sources:
            try:
                await self._navigate(session, source.url, site)
                page = await self._extract(session, source.url, detail=True)
                if page.get("text"):
                    source.excerpt = privacy.sanitize(page["text"])[:2200]
                    source.detail_read = True
            except BrowserAccessError as exc:
                if exc.code in {"login_required", "challenge"}:
                    break
                # 搜索摘要仍可展示，但不能声称已阅读正文。
                continue
            except TimeoutError:
                # 一个详情页超时不能抹掉已经核实的搜索结果及之前读到的正文。
                break
        return WebResearchResult("ok" if sources else "empty", query=query, site=site, sources=sources)
