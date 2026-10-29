"""本机浏览器检索检查；不写业务数据库，不打印其他标签页或网页正文。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.core.enums import RiskLevel
from app.services.browser_mcp import ChromeResearchClient
from app.services.web_access import WebAccessService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", help="执行一次真实检索，例如：帮我在小红书搜索考试焦虑的同学经历")
    args = parser.parse_args()
    settings = Settings()
    if args.search:
        from app.services.browser_worker import stop_browser_worker
        try:
            result = WebAccessService(settings).search(settings.web_access_username, args.search, [], RiskLevel.LOW)
        finally:
            stop_browser_worker()
        print(json.dumps({"status": result.status, "message": result.message, "query": result.query,
                          "sourceCount": len(result.sources), "detailReadCount": sum(s.detail_read for s in result.sources),
                          "toolCalls": result.calls}, ensure_ascii=False, indent=2))
        return 0 if result.status == "ok" else 1

    async def probe():
        async with asyncio.timeout(settings.web_access_timeout_seconds):
            async with ChromeResearchClient(settings)._session():
                print("MCP 握手与只读检索所需工具已就绪；尚未验证 Chrome 授权或网站登录状态。")
    try:
        asyncio.run(probe())
        return 0
    except Exception as exc:
        print(f"MCP 检查失败：{type(exc).__name__}。请检查 Node 版本与 WEB_ACCESS_MCP_* 配置。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
