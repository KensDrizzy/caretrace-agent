from __future__ import annotations

import asyncio
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.enums import RiskLevel
from app.schemas.dtos import AiMessage
from app.services.browser_mcp import BrowserAccessError, ChromeResearchClient, safe_public_url
from app.services.web_access import (
    WebAccessService, WebResearchResult, WebSource, requested_site, topic_query,
    wants_web_search, web_response_text,
)


@pytest.fixture(autouse=True)
def isolated_browser_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))


def settings(**kwargs):
    return Settings(_env_file=None, web_access_enabled=True, web_access_username="student", web_access_persistent=False, **kwargs)


@pytest.mark.parametrize("text", [
    "帮我去小红书搜索和我一样的同学", "帮我在知乎看看类似经历", "在网上搜一下考试焦虑的经验",
    "帮我在 https://www.douban.com 搜索类似的经历", "帮我找几个和我相似的帖子",
])
def test_explicit_web_intent(text):
    assert wants_web_search(text)


@pytest.mark.parametrize("text", ["我最近考试很焦虑", "我在小红书看到了不开心的事情", "不要帮我在网上搜索", "别搜小红书", "解释一下二分查找"])
def test_no_implicit_or_negated_browsing(text):
    assert not wants_web_search(text)


def test_contextual_query_is_anonymous_and_uses_recent_topic():
    history = [AiMessage(role="user", content="我叫张三，浙江工业大学，手机号13812345678，研究生论文写不出来"),
               AiMessage(role="assistant", content="要不要聊聊你的失恋？")]
    query = topic_query("帮我在小红书搜和我一样的同学", history)
    assert query == "研究生 论文压力 经历分享"
    assert all(value not in query for value in ["张三", "浙江工业大学", "13812345678", "失恋"])
    assert topic_query("帮我搜考试焦虑", history) == "大学生 考试焦虑 焦虑情绪 经历分享"
    assert topic_query("帮我找相似经历", []) == ""


def test_site_resolution_and_private_targets():
    assert requested_site("帮我去小红书搜") == "xiaohongshu.com"
    assert requested_site("帮我在 www.douban.com 搜") == "douban.com"
    assert requested_site("帮我在 https://example.org 搜") == "example.org"
    for url in ["http://127.0.0.1/xiaohongshu.com", "https://localhost/xiaohongshu", "https://user:pass@example.org", "https://example.org:8000"]:
        with pytest.raises(ValueError):
            requested_site("帮我搜索 " + url)


@pytest.mark.parametrize("username,risk,status", [("another_student", RiskLevel.LOW, "not_bound"),
                                                  ("student", RiskLevel.HIGH, "high_risk")])
def test_unauthorized_or_high_risk_never_connects(monkeypatch, username, risk, status):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not connect")
    monkeypatch.setattr(ChromeResearchClient, "search", forbidden)
    result = WebAccessService(settings()).search(username, "帮我在小红书搜考试焦虑", [], risk)
    assert result.status == status


def test_disabled_and_missing_topic_do_not_launch_browser(monkeypatch):
    monkeypatch.setattr(ChromeResearchClient, "search", lambda *args: pytest.fail("must not connect"))
    disabled = settings()
    disabled.web_access_enabled = False
    assert WebAccessService(disabled).search("student", "帮我搜考试焦虑", [], RiskLevel.LOW).status == "disabled"
    assert WebAccessService(settings()).search("student", "帮我搜类似经历", [], RiskLevel.LOW).status == "needs_topic"


@pytest.mark.parametrize("endpoint", ["http://example.org:9222", "http://127.0.0.1@evil.com", "http://localhost/path", "file:///tmp/a"])
def test_cdp_endpoint_cannot_be_supplied_by_remote_host(endpoint):
    with pytest.raises(BrowserAccessError):
        ChromeResearchClient(settings(web_access_browser_url=endpoint)).server_parameters()


def test_mcp_environment_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("DATABASE_URL", "test-db-secret")
    params = ChromeResearchClient(settings()).server_parameters()
    assert "OPENAI_API_KEY" not in params.env
    assert "DATABASE_URL" not in params.env
    assert "--autoConnect" in params.args


def test_url_guard_blocks_internal_dns_and_account_operations(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(BrowserAccessError):
        safe_public_url("https://public.example.com/article")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, 1, 6, "", ("8.8.8.8", 443))])
    for url in ["https://example.com/logout", "https://example.com/user/profile", "https://example.com/article?redirect=https://evil.com", "https://evil.com/post"]:
        with pytest.raises(BrowserAccessError):
            safe_public_url(url, "example.com")
    assert safe_public_url("https://www.example.com/article", "example.com")


def test_sources_cannot_be_invented_by_model():
    payload = WebResearchResult("ok", sources=[WebSource("考试压力分享", "https://example.com/post", "考试时紧张")]).payload()
    text = web_response_text("参考[假的](https://evil.com)和 https://fake.com [9] [1]。", payload)
    assert "https://example.com/post" in text
    assert "evil.com" not in text and "fake.com" not in text and "[9]" not in text
    assert "仅搜索结果摘要" in text
    assert web_response_text("我找到很多同学", {"status": "login_required"}).startswith("这个网站需要登录")


@pytest.mark.parametrize("target,status", [
    ("https://evil.example.org/explore/abc", "blocked"),
    ("http://www.xiaohongshu.com/explore/abc", "blocked"),
    ("https://www.xiaohongshu.com:8443/explore/abc", "blocked"),
    ("https://www.xiaohongshu.com/explore/another", "blocked"),
    ("https://www.xiaohongshu.com/user/profile/abc", "login_required"),
])
def test_redirect_guard_runs_before_reading_any_document(target, status):
    import shutil
    import subprocess
    from app.services.browser_mcp import extraction_script

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser integration")
    script = extraction_script("xiaohongshu.com", detail=True, paths=["/explore/abc"])
    code = ("global.location = new URL(" + json.dumps(target) + ");"
            "global.document = new Proxy({}, {get() {throw new Error('must not read redirected document')}});"
            "console.log(JSON.stringify((" + script + ")()));")
    result = subprocess.run([node, "-e", code], capture_output=True, text=True, check=True, timeout=5)
    assert json.loads(result.stdout) == {"access": status}


def test_xiaohongshu_prefers_search_link_with_public_access_token():
    import shutil
    import subprocess
    from app.services.browser_mcp import extraction_script

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser integration")
    public_link = "https://www.xiaohongshu.com/search_result/abc?xsec_token=public-link-token"
    script = extraction_script("xiaohongshu.com", detail=False, paths=["/search_result"])
    code = """
      global.location = new URL('https://www.xiaohongshu.com/search_result?keyword=exam');
      const title = {innerText: '我的备考经历'};
      const card = {querySelector: () => title, querySelectorAll: () => [
        {href: 'https://www.xiaohongshu.com/explore/abc'}, {href: PUBLIC_LINK}
      ]};
      global.document = {title: '小红书', querySelector: () => null,
        querySelectorAll: selector => selector === '.note-item' ? [card] : []};
    """.replace("PUBLIC_LINK", json.dumps(public_link))
    result = subprocess.run([node, "-e", code + "console.log(JSON.stringify((" + script + ")()));"],
                            capture_output=True, text=True, check=True, timeout=5)
    assert json.loads(result.stdout)["links"][0]["url"] == public_link


class FakeSession:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "new_page":
            text = "# new_page response\n## Pages\n0: https://private.example.org\n7: about:blank [selected]"
        elif name == "evaluate_script":
            if self.failure:
                data = {"access": self.failure}
            elif "const detail = true" in arguments["function"]:
                data = {"access": "ok", "text": "我也有考试压力，和朋友聊过后慢慢调整了学习安排。"}
            else:
                data = {"access": "ok", "links": [
                    {"title": "我的备考经历", "url": "https://www.xiaohongshu.com/explore/abc123?xsec_token=public-link-token", "excerpt": "备考时焦虑"},
                    {"title": "个人主页", "url": "https://www.xiaohongshu.com/user/profile/xyz", "excerpt": "不能读取"},
                ]}
            text = "Script ran on page and returned:\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"
        else:
            text = "OK"
        return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text=text)])


def fake_browser(monkeypatch, failure=None):
    monkeypatch.setattr("app.services.browser_mcp.safe_public_url", lambda url, domain="": url)
    session = FakeSession(failure)
    @asynccontextmanager
    async def connection(self):
        yield session
    monkeypatch.setattr(ChromeResearchClient, "_session", connection)
    return session


def test_browser_reads_only_own_page_and_closes_it(monkeypatch):
    session = fake_browser(monkeypatch)
    client = ChromeResearchClient(settings())
    result = asyncio.run(client.search("大学生 考试焦虑 经历分享", "xiaohongshu.com"))
    assert result.status == "ok" and len(result.sources) == 1
    assert result.sources[0].detail_read
    assert session.calls[-1] == ("close_page", {"pageId": 7})
    assert all(args.get("pageId") == 7 for name, args in session.calls if name != "new_page")
    assert not any(name in {"click", "fill", "list_pages"} for name, _ in session.calls)
    assert "private.example.org" not in json.dumps(client.calls)
    assert all(args["handleBeforeUnload"] == "dismiss" for name, args in session.calls if name == "navigate_page")


@pytest.mark.parametrize("failure", ["login_required", "challenge", "blocked"])
def test_browser_failure_preserves_status_and_closes_only_owned_tab(monkeypatch, failure):
    session = fake_browser(monkeypatch, failure)
    result = WebAccessService(settings()).search("student", "帮我在小红书搜考试焦虑经历", [], RiskLevel.LOW)
    assert result.status == failure and not result.sources
    assert session.calls[-1] == ("close_page", {"pageId": 7})


def test_runtime_waits_for_web_and_reviews_final_sources(monkeypatch):
    from app.agent_eval.testkit import run_case, default_eval_settings
    from app.agent_eval.schemas import GoldenCase, ConversationTurn, ExpectedSpec, ModelScript
    from app.agent_eval.invariant_evaluator import InvariantEvaluator
    calls = []
    def search(self, username, text, history, risk):
        calls.append((text, risk))
        return WebResearchResult("ok", sources=[WebSource("备考分享", "https://www.xiaohongshu.com/explore/abc", "备考压力")])
    monkeypatch.setattr(WebAccessService, "search", search)
    case = GoldenCase(caseId="web-runtime", source="synthetic", labelStatus="gold", conversation=[
        ConversationTurn(role="user", content="最近我考试压力很大"),
        ConversationTurn(role="user", content="帮我去小红书找找和我一样的同学")],
        expected=ExpectedSpec(intent=["CHAT", "CONSULT"], risk=["LOW"]),
        modelScript=ModelScript(intent="CHAT", risk="LOW", response="这篇分享谈到了备考时的压力，可以参考其中调整学习节奏的经验 [1]。"))
    trace = run_case(case, default_eval_settings())
    assert len(calls) == 1
    assert "https://www.xiaohongshu.com/explore/abc" in trace.finalResponse
    accepted = trace.artifact_by_id(trace.finalResponseArtifactId)
    assert accepted.payload["text"] == trace.finalResponse
    assert InvariantEvaluator().evaluate(case, trace).passed
    kinds = trace.event_types()
    assert "WEB_RESEARCH_COMPLETED" in kinds
    assert kinds.index("WEB_RESEARCH_COMPLETED") < kinds.index("FINAL_RESPONSE_GENERATED")


def test_runtime_failed_search_cannot_hallucinate_results(monkeypatch):
    from app.agent_eval.testkit import run_case
    from app.agent_eval.schemas import GoldenCase, ConversationTurn, ExpectedSpec, ModelScript
    monkeypatch.setattr(WebAccessService, "search", lambda *args: WebResearchResult("login_required", "这个网站需要登录。"))
    case = GoldenCase(caseId="web-login", source="synthetic", labelStatus="gold", conversation=[ConversationTurn(role="user", content="帮我在小红书搜考试焦虑")],
        expected=ExpectedSpec(intent=["CONSULT", "CHAT"], risk=["LOW"]),
        modelScript=ModelScript(risk="LOW", response="我找到了100个同学的真实案例。"))
    trace = run_case(case)
    assert trace.finalResponse == "这个网站需要登录。"
    assert "100" not in trace.finalResponse


def test_real_stdio_mcp_transport(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.browser_mcp.safe_public_url", lambda url, domain="": url)
    log = tmp_path / "calls.jsonl"
    config = settings(web_access_mcp_command=sys.executable,
                      web_access_mcp_args=[str(Path(__file__).with_name("browser_mcp_fixture.py")), "--log", str(log)])
    result = asyncio.run(ChromeResearchClient(config).search("大学生 考试焦虑 经历分享", "xiaohongshu.com"))
    assert result.status == "ok" and result.sources[0].detail_read
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[-1] == {"name": "close_page", "pageId": 8}
    assert all(c.get("pageId") == 8 for c in calls[1:])


def test_busy_browser_rejects_second_search(monkeypatch):
    import fcntl
    session = fake_browser(monkeypatch)
    directory = Path.home() / ".cache/caretrace"
    directory.mkdir(parents=True)
    with (directory / "browser-research.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = WebAccessService(settings()).search("student", "帮我搜考试焦虑帖子", [], RiskLevel.LOW)
        assert result.status == "busy" and not session.calls


def test_navigation_timeout_still_closes_page(monkeypatch):
    session = fake_browser(monkeypatch)
    original = session.call_tool
    async def fail(name, arguments):
        if name == "navigate_page":
            raise TimeoutError()
        return await original(name, arguments)
    session.call_tool = fail
    result = WebAccessService(settings()).search("student", "帮我搜考试焦虑帖子", [], RiskLevel.LOW)
    assert result.status == "timeout"
    assert session.calls[-1] == ("close_page", {"pageId": 7})


def test_detail_timeout_preserves_verified_listing_and_discards_connection(monkeypatch):
    session = fake_browser(monkeypatch)
    original = session.call_tool

    async def fail_detail(name, arguments):
        if name == "navigate_page" and "/explore/" in arguments["url"]:
            raise TimeoutError()
        return await original(name, arguments)

    session.call_tool = fail_detail
    client = ChromeResearchClient(settings())
    result = asyncio.run(client.search("考试压力", "xiaohongshu.com"))
    assert result.status == "ok" and len(result.sources) == 1
    assert not result.sources[0].detail_read
    assert client.reset_required
    assert session.calls[-1] == ("close_page", {"pageId": 7})


def test_bing_redirect_is_decoded_locally():
    import base64
    from app.services.browser_mcp import result_url
    url = "https://www.douban.com/group/topic/123"
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    assert result_url("https://www.bing.com/ck/a?u=a1" + encoded) == url


def test_persistent_owner_reuses_stdio_session_and_shuts_down(monkeypatch, tmp_path):
    from app.services.browser_worker import BrowserResearchWorker
    monkeypatch.setattr("app.services.browser_mcp.safe_public_url", lambda url, domain="": url)
    log = tmp_path / "persistent.jsonl"
    config = settings(web_access_mcp_command=sys.executable,
                      web_access_mcp_args=[str(Path(__file__).with_name("browser_mcp_fixture.py")), "--log", str(log)])
    worker = BrowserResearchWorker(config)
    try:
        first = worker.search("大学生 考试焦虑 经历分享", "xiaohongshu.com")
        second = worker.search("大学生 宿舍关系 经历分享", "xiaohongshu.com")
        assert first.status == second.status == "ok"
        assert first.calls is not second.calls
    finally:
        worker.stop()
    assert not worker.thread.is_alive()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["name"] for c in calls].count("close_page") == 2


@pytest.mark.parametrize("failure", [TimeoutError(), BrowserAccessError("unavailable")])
def test_persistent_owner_reopens_connection_after_transport_failure(monkeypatch, failure):
    from app.services.browser_worker import BrowserResearchWorker
    opened, closed, searched = [], [], []

    @asynccontextmanager
    async def connection(self):
        session = object()
        opened.append(session)
        try:
            yield session
        finally:
            closed.append(session)

    async def research(self, session, query, site):
        searched.append(session)
        if len(searched) == 1:
            raise failure
        return WebResearchResult("empty")

    monkeypatch.setattr(ChromeResearchClient, "_session", connection)
    monkeypatch.setattr(ChromeResearchClient, "research_in_session", research)
    worker = BrowserResearchWorker(settings())
    try:
        assert worker.search("考试压力", "xiaohongshu.com").status in {"timeout", "unavailable"}
        assert worker.search("考试压力", "xiaohongshu.com").status == "empty"
        assert len(opened) == 2 and searched[0] is not searched[1]
        assert opened[0] in closed
    finally:
        worker.stop()
    assert closed == opened and not worker.thread.is_alive()
