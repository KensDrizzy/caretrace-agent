# 学生对话中的网页经历检索

学生可以直接说：

- “最近论文压力很大，帮我在小红书上找找其他同学的经历。”
- 先聊考试压力，再说：“帮我在小红书搜索和我当前情况相似的同学。”
- “去知乎搜一下宿舍关系的经验。”
- “帮我在 douban.com 搜索大学生失眠的经历。”

返回内容包含概括、公开来源链接，以及“已读取正文”或“仅搜索结果摘要”的说明。只能寻找公开的相似主题分享，不识别或联系具体学生，不据帖子推断作者病情。

## 本机演示配置

需要 Python 3.11+、已安装的 `requirements.txt`、Node 22.12+ 和支持远程调试的 Chrome。Chrome DevTools MCP 固定为经过本项目协议校验的 1.12.0；使用官方 npm registry，不修改机器全局 npm 配置。

在 `.env` 中添加：

```dotenv
WEB_ACCESS_ENABLED=true
WEB_ACCESS_USERNAME=student
WEB_ACCESS_BROWSER_URL=
WEB_ACCESS_MCP_COMMAND=npx
WEB_ACCESS_MCP_ARGS=["--yes","chrome-devtools-mcp@1.12.0"]
# macOS Homebrew Node 22；其他环境填写实际 bin 目录或留空使用 PATH
WEB_ACCESS_NODE_BIN=/opt/homebrew/opt/node@22/bin
WEB_ACCESS_TIMEOUT_SECONDS=90
WEB_ACCESS_TOOL_TIMEOUT_SECONDS=20
WEB_ACCESS_MAX_RESULTS=3
WEB_ACCESS_PERSISTENT=true
```

`WEB_ACCESS_USERNAME` 必须是要使用你本机浏览器的 CareTrace 登录名，留空或其他账号都不能访问。默认关闭能力；这是单人本机演示配置，不是多学生远程浏览器授权方案。

1. 在本机 Chrome 中登录小红书或目标网站。
2. 打开 `chrome://inspect/#remote-debugging` 并开启远程调试。首次连接若 Chrome 弹出授权窗口，由使用者点击允许。
3. 如果 MySQL/Redis 尚未启动，先运行 `docker compose up -d mysql redis`。再启动 `bash scripts/run-local-web.sh`，打开 `http://127.0.0.1:8081/student.html`。
4. 用绑定的学生账号登录，先描述困扰，再请求搜索。

空的 `WEB_ACCESS_BROWSER_URL` 会优先读取本机 Chrome 的 `DevToolsActivePort` 文件并使用 WS 连接；没有发现文件时使用 `--autoConnect`。也可显式指定 `http://127.0.0.1:9222` 或本机 `ws://127.0.0.1:端口/devtools/browser/ID`。不支持远程 CDP 地址。

本地启动入口使用原有 `.env` 数据库配置，不修改 Docker。若数据库在 Compose 中，应使用映射到宿主机的端口，例如 MySQL 13306、Redis 16379。后端不能运行在 Docker 中再把 `127.0.0.1` 当成宿主机 Chrome；本方案选择本机后端解决这一边界。

## 实现链路

1. 显式联网意图使 Coordinator 为 ContextAgent 派发上下文任务。风险评估完成前不执行联网。
2. ContextAgent 按需加载 `skills/web-access/SKILL.md`；站点策略与匿名主题词来自其 `references/search-policy.json`。
3. 从当前输入或最近的用户消息选择宽泛主题，构成“大学生/研究生＋主题＋经历分享”。不向搜索站点发送姓名、学校、联系方式或咨询全文。主题不明确则请学生补充。
4. ChromeResearchClient 经 MCP 创建自己的标签页；小红书/知乎使用站内搜索，其他网站通过 Bing `site:` 检索。首次结果页最多提取 20 个候选，最多阅读配置数量的详情页，上限 5。
5. 结果作为 `context.webResearch` 提供给 ResponseAgent。网页文本是外部不可信数据；链接来自实际检索结果，由代码附加到候选文本，然后 SafetyAgent 审核同一份完整回复。
6. 来源链接随助手消息存入会话历史。`WEB_RESEARCH_COMPLETED/UNAVAILABLE` 记录状态、泛化查询词、来源数及 MCP 操作耗时，不保存 MCP 返回的其他标签页列表。

## 工具边界与失败处理

- 仅允许 `new_page`、`navigate_page`、固定 DOM 读取脚本和关闭本次创建的页；不允许模型生成 JavaScript、读取 cookie、操作已有标签页、发帖、私信或点赞。
- 导航前验证 HTTPS 域名和公开 DNS，过滤内网、账号及动作链接；DOM 读取前校验最终域名、路径和登录状态。1.12.0 已移除旧版 `allowList`，这里不拦截网络层重定向或子资源，不是任意恶意网站的网络沙箱。使用已知的公开站点进行本机演示。
- 默认由后台线程中的单一 owner 协程持有 MCP 长连接，首次连接后复用 Chrome 授权；应用关闭时释放。超时或连接失败时销毁 MCP 进程，下次请求重建，避免被旧请求占住队列。进程重启或浏览器断开后可能再次要求 Chrome 确认。`WEB_ACCESS_PERSISTENT=false` 用于单次诊断。
- 持久连接持有跨进程文件锁，重复/并发请求返回忙碌提示，避免共享登录态冲突。本机演示使用一个 Uvicorn worker。
- MCP session 在同一个 owner 协程内创建、使用和清理，设置单工具和总超时；首次创建标签页给 Chrome 授权预留最多 60 秒。正常及异常退出均尝试关闭本次已知的标签页。连接创建超时尚未取得页 ID 时，不猜测或关闭用户标签页。
- 登录、验证码、无结果、未绑定和连接失败均明确告知学生，不让模型编造检索成功。详情页受限时保留已有摘要，并明确没有核实正文。
- 小红书优先保留搜索卡片中带公开访问参数的笔记链接，避免误用不带参数的封面链接。单个详情页超时保留已经取得的来源和正文，并回收发生超时的 MCP 连接。
- 高风险场景不启动相似经历搜索，继续现有安全支持流程。

## 检查

```bash
# 仅检查 MCP 协议与工具 schema，不操作 Chrome
.venv/bin/python scripts/check_web_access.py

# 真实连接本机 Chrome 搜索；只打印状态和计数，不写业务数据库
.venv/bin/python scripts/check_web_access.py --search '帮我在小红书搜索考试焦虑的同学经历'

# 自动测试使用 fake 页面与真实 stdio 测试服务，不使用个人登录态
.venv/bin/python -m pytest -q tests/test_web_access.py
```

Chrome 的授权和网站登录/验证码必须由本机使用者完成。出现连接失败时先检查 Chrome，再确认 Node 路径。更多连接参数参见 [Chrome DevTools MCP 官方文档](https://github.com/ChromeDevTools/chrome-devtools-mcp)。

如果已允许调试但一直卡在 `new_page`，检查是否仍配置了旧版 `chrome-devtools-mcp@0.26.0`。本机实测该版本会在初始化已有标签页时耗时过长；1.12.0 按需初始化页面，已完成真实创建/关闭标签页及小红书检索验证。修改版本后需重启本机后端。

### 本机验证记录（2026-10-29）

- 从学生页面提交“帮我在小红书搜索大学生考试焦虑的经历”，实际返回 3 篇公开笔记，3 篇均成功读取正文；页面展示来源链接和“已读取正文”标记。
- 对应 trace：`95bf422116614ae289bd80d8cbd28f89`，事件 `WEB_RESEARCH_COMPLETED`；创建、搜索、详情读取、关闭标签页共 10 次 MCP 调用全部成功。
- 本提交的独立快照全量测试 `108 passed`，其中网页检索相关 `42 passed`。这些测试使用隔离的模拟页面；真实网站结果来自上述单独的人工联调，后续仍可能受网站登录状态、验证或网络状况影响。
