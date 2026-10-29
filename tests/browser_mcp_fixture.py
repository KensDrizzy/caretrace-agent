"""真实 stdio 协议测试服务；只返回固定页面，不连接浏览器或互联网。"""
import argparse
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    args, _ = parser.parse_known_args()
    server = FastMCP("caretrace-browser-fixture")

    def record(name, values):
        with Path(args.log).open("a") as output:
            output.write(json.dumps({"name": name, **values}) + "\n")

    @server.tool()
    def new_page(url: str, background: bool = True) -> str:
        record("new_page", {"url": url})
        return "## Pages\n1: https://unrelated.example.com\n8: about:blank [selected]"

    @server.tool()
    def navigate_page(pageId: int, url: str, type: str = "url", timeout: int = 1000, handleBeforeUnload: str = "dismiss") -> str:
        record("navigate_page", {"pageId": pageId, "url": url})
        return "OK"

    @server.tool()
    def evaluate_script(pageId: int, function: str, dialogAction: str = "dismiss", waitForStableDom: bool = False) -> str:
        record("evaluate_script", {"pageId": pageId})
        if "const detail = true" in function:
            data = {"access": "ok", "text": "考试压力很大时，我和朋友聊了聊学习安排。"}
        else:
            data = {"access": "ok", "links": [{"title": "我的备考经历", "url": "https://www.xiaohongshu.com/explore/abc", "excerpt": "备考压力"}]}
        return "```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"

    @server.tool()
    def close_page(pageId: int) -> str:
        record("close_page", {"pageId": pageId})
        return "OK"

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
