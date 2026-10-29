"""一个后台线程、一个 owner 协程持有 MCP 会话，跨对话复用 Chrome 授权。"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from app.services.browser_mcp import ChromeResearchClient, browser_error_code, browser_lease
from app.services.web_access import WebResearchResult


class BrowserResearchWorker:
    def __init__(self, settings):
        self.settings = settings
        self.client = ChromeResearchClient(settings)
        self.gate = threading.Lock()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="caretrace-browser", daemon=True)
        self.loop = None
        self.queue = None
        self.task = None
        self.thread.start()

    def search(self, query: str, site: str) -> WebResearchResult:
        if not self.gate.acquire(blocking=False):
            return WebResearchResult("busy")
        try:
            if not self.ready.wait(5) or not self.thread.is_alive():
                return WebResearchResult("unavailable")
            future = concurrent.futures.Future()
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (query, site, future))
            try:
                return future.result(timeout=self.settings.web_access_timeout_seconds + 10)
            except concurrent.futures.TimeoutError:
                self.stop()
                return WebResearchResult("timeout")
        finally:
            self.gate.release()

    def _run(self):
        try:
            asyncio.run(self._serve())
        except asyncio.CancelledError:
            pass

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        self.task = asyncio.current_task()
        self.ready.set()
        job = None
        try:
            while True:
                job = await self.queue.get()
                self.client.calls = []
                try:
                    with browser_lease():
                        # 同一个 owner 协程进入/退出所有 AnyIO context；绝不跨请求
                        # 或跨 Task 关闭 MCP session。空闲时继续持有连接与浏览器租约。
                        async with asyncio.timeout(self.settings.web_access_timeout_seconds):
                            # 初始化的超时仅覆盖握手，后续每个检索有独立预算。
                            session_context = self.client._session()
                            session = await session_context.__aenter__()
                        try:
                            while True:
                                query, site, future = job
                                try:
                                    async with asyncio.timeout(self.settings.web_access_timeout_seconds):
                                        result = await self.client.research_in_session(session, query, site)
                                except Exception as exc:
                                    result = WebResearchResult(browser_error_code(exc))
                                result.calls = list(self.client.calls)
                                if not future.done():
                                    future.set_result(result)
                                # MCP cancellation does not necessarily cancel the server's
                                # in-flight CDP operation. Dispose the stuck process before
                                # accepting another job instead of queuing behind its mutex.
                                if result.status in {"timeout", "unavailable"} or self.client.reset_required:
                                    break
                                job = await self.queue.get()
                        finally:
                            await session_context.__aexit__(None, None, None)
                except Exception as exc:
                    if job and not job[2].done():
                        job[2].set_result(WebResearchResult(browser_error_code(exc)))
                    job = None
        finally:
            if job and not job[2].done():
                job[2].set_result(WebResearchResult("unavailable"))

    def stop(self):
        if self.loop is not None and self.thread.is_alive():
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:
                pass
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=5)


_worker = None
_worker_lock = threading.Lock()


def get_browser_worker(settings) -> BrowserResearchWorker:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.thread.is_alive():
            _worker = BrowserResearchWorker(settings)
        return _worker


def stop_browser_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None
