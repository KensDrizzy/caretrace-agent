# CareTrace Python

## 核心能力

- 学生端 SSE 流式聊天，前端可展示打字机式输出。
- Basic Auth 登录，支持学生和管理员角色隔离。
- 事件驱动多 Agent 协作 runtime：Coordinator、Understanding、Safety、Context、Response 通过共享黑板、任务认领和安全审查协作。
- 动态路由 RAG：先判断 `CHAT / CONSULT / RISK`，普通问题不查知识库，咨询和风险场景才进入检索增强。
- Chroma 向量 RAG 知识库：支持 Markdown、txt、PDF 文件上传，自动切块，使用 `text-embedding-3-small` 写入向量库，并与 BM25 关键词召回融合后进入本地 reranker；向量不可用时保留本地 BM25 + 词面检索兜底。
- 心理风险评估：高风险词典优先、LLM JSON 评估、关键词兜底。
- 后台报告：记录情绪标签、情绪分数、风险等级、置信度和摘要，但学生端不展示后台评估结果。
- 数据闭环：咨询/风险消息完整写入 MySQL，短期上下文写入 Redis，高风险消息写入 Excel 台账并通过邮件发送预警。
- 本地微调模型接入：支持通过 Ollama 加载 `mindbridge-qwen2.5-7b-ft-q4_k_m.gguf`。
- OpenAI-compatible API 接入：也可切换到云端模型。
- MCP 工具服务：暴露 Excel 报告写入和风险通知工具，后端高风险后处理通过 MCP client 调用这些工具。
- Agent Trace v2：每次请求一条统一 `trace_id` 的可回放轨迹，覆盖 Agent 决策、任务调度、Artifact、LLM、RAG、工具执行和最终回复；事件落 `agent_trace_events` 表。
- 离线评测：Golden Dataset（JSONL）+ 确定性 Hard Gate（轨迹不变量）+ 可替换 Rubric LLM Judge + 批量回归报告。
- RAG 评测：Recall@K、Precision@K、MRR、NDCG@K、HitRate。

## 技术栈

```text
语言：Python
Web 框架：FastAPI
服务运行：Uvicorn / ASGI
数据库：MySQL，SQLAlchemy ORM，PyMySQL 驱动
短期记忆：Redis
配置管理：pydantic-settings，.env
AI 接入：Ollama，本地微调 GGUF 模型，OpenAI-compatible API，Mock Provider
Agent 编排：事件驱动黑板协作 runtime
RAG：本地知识库切块、OpenAI Embeddings、Chroma 向量库、BM25、分数融合、本地 reranker、上下文扩展
流式输出：Server-Sent Events
文档解析：pypdf
Excel 台账：openpyxl
邮件预警：SMTP / smtplib
前端：原生 HTML / CSS / JavaScript
认证：Basic Auth
工具协议：MCP
评测与测试：自研 app/agent_eval（Hard Gate + Rubric Judge）、pytest、unittest
```

说明：当前 Python 版只保留事件驱动多 Agent runtime，入口在 `app/agents/event_driven_runtime.py`。共享返回类型定义在 `app/agents/result.py`。RAG 默认使用 Chroma 本地持久化向量库做语义召回，同时用 BM25 做关键词召回，再融合并本地 rerank；未安装 Chroma、未配置 `OPENAI_API_KEY` 或向量服务异常时，会自动回退到本地 BM25 + `hybrid_score` reranker，避免演示环境中断。

## 目录结构

```text
app/
├── agent_eval/      # 离线评测：数据集加载、Trace 解析、Hard Gate、Rubric Judge、回放 testkit、CLI
├── agents/          # 事件驱动多 Agent runtime
├── api/             # FastAPI 路由
├── core/            # 配置、数据库、安全、启动初始化
├── harness/         # 工程 harness（mock AI + 临时 SQLite 的全链路自检）
├── knowledge/       # 内置校园心理知识库
├── mcp_tools/       # MCP 工具服务
├── models/          # SQLAlchemy 实体
├── rag_eval/        # RAG 评测脚本和数据集
├── schemas/         # Pydantic DTO
├── services/        # AI、聊天、知识库、评估、报告、工具、Trace 服务
└── static/          # 原生前端页面

models/mindbridge-qwen2.5-7b-ft/
├── Modelfile        # Ollama 模型定义
└── README.md        # GGUF 模型放置说明

skills/              # 标准 Skill 包（*/SKILL.md，运行时加载）
tests/
├── eval/            # Golden Dataset（caretrace_gold.jsonl）与生成脚本
└── test_*.py        # unittest 回归 + pytest Trace/评测场景
reports/             # 离线评测报告输出（eval_report.json）

scripts/
├── run-dev.sh
├── start-ollama.sh
├── create-finetuned-model.sh
├── package-release.sh
├── migrate_trace_v2.py   # Trace v2 数据库迁移（幂等）
└── smoke_trace_v2.py     # Trace v2 冒烟脚本
```

## Agent loop

每轮对话默认进入事件驱动多 Agent 协作 runtime。Coordinator 维护共享黑板和任务板，专业 Agent 根据能力和置信度认领任务，发布 artifact，再由安全审查和最终采纳机制收敛输出：

```text
TURN_STARTED
-> CoordinatorAgent 创建任务
-> UnderstandingAgent / SafetyAgent / ContextAgent 认领任务并发布 artifact
-> ResponseAgent 调用模型生成完整候选文本（response_candidate artifact）
-> SafetyAgent 审核候选文本（规则 + LLM 语义审核，引用 artifact id + version）
-> CoordinatorAgent 校验审核通过后 FINAL_ACCEPTED
-> ChatService 逐块推送同一份已审核文本（不再二次调用模型）
-> 工具执行（Excel / 个案 / 预警，幂等）
-> TRACE_COMPLETED，Trace v2（含全部事件与指标）落库
```

各 Agent 分工：

- `CoordinatorAgent`：维护任务板、预算、安全门槛、冲突仲裁和最终采纳。
- `UnderstandingAgent`：判断 `CHAT / CONSULT / RISK`，发布 intent artifact。
- `SafetyAgent`：独立评估风险，必要时发布 `SAFETY_OVERRIDE`，并审核 ResponseAgent 生成的候选文本（审核失败创建修订任务，未审核/审核未通过的文本绝不会被发送）。
- `ContextAgent`：按需聚合 Redis / MySQL 记忆、RAG 检索结果和 Skill 约束。
- `ResponseAgent`：根据黑板 artifact 调用模型生成完整候选回复文本；LLM 不可用时降级为确定性安全兜底文本。

## 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含：

```text
chromadb
pymysql
redis
```

`AGENT_FRAMEWORK` 仍会读取环境变量，但当前只支持 `event_driven_multi_agent`。历史值或未知值会在状态接口中标记为 fallback，并实际使用事件驱动 runtime。

## MySQL 和 Redis 配置

系统默认使用 MySQL 保存完整业务数据和完整聊天消息，使用 Redis 保存短期对话记忆。启动服务前先创建数据库：

```sql
CREATE DATABASE mindbridge DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'mindbridge'@'%' IDENTIFIED BY 'mindbridge';
GRANT ALL PRIVILEGES ON mindbridge.* TO 'mindbridge'@'%';
FLUSH PRIVILEGES;
```

`.env` 中配置连接：

```env
DATABASE_URL=mysql+pymysql://mindbridge:mindbridge@127.0.0.1:3306/mindbridge?charset=utf8mb4
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_MEMORY_TTL_SECONDS=86400
REDIS_MEMORY_MAX_MESSAGES=40
```

完整聊天记录写入 MySQL 的 `chat_sessions`、`chat_messages` 等表。Redis 只保存每个会话最近 `REDIS_MEMORY_MAX_MESSAGES` 条短期上下文，并通过 `REDIS_MEMORY_TTL_SECONDS` 自动过期。

## 默认账号

```text
student / student123
admin / admin123
```

## Docker Compose 一键启动

仓库提供 `Dockerfile` 和 `docker-compose.yml`，会启动：

- `mysql`：MySQL 8.4，容器内端口 `3306`，宿主机映射 `13306`
- `redis`：Redis 7，容器内端口 `6379`，宿主机映射 `16379`
- `app`：CareTrace FastAPI 服务，宿主机端口 `8080`

默认配置会让应用容器访问宿主机 Ollama：

```bash
docker compose up -d --build
```

如果 Ollama 已经有下列模型，容器即可使用真实本地聊天模型链路：

```text
mindbridge-qwen2.5-7b-ft:latest
```

## Chroma 向量库与快照

应用启动时会同步 `app/knowledge/*.md` 内置默认知识库到数据库。当前默认文档覆盖校园心理支持总则、风险等级策略、焦虑恐慌、情绪低落、睡眠作息、学业压力、考试季、人际关系、新生适应、咨询转介和隐私边界等主题；如果默认 md 内容发生变化，重启后对应来源会按当前切块规则刷新入库。

知识库默认优先使用 Chroma 持久化向量库，embedding 由 OpenAI `text-embedding-3-small` 提供。查询时会同时取向量候选和 BM25 候选，按配置权重融合后进入本地 reranker。没有 `OPENAI_API_KEY`、缺少 `chromadb` 或向量调用失败时，会回退到本地 BM25 + `hybrid_score` reranker：

```env
OPENAI_API_KEY=你的_API_Key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
KNOWLEDGE_VECTOR_ENABLED=true
KNOWLEDGE_VECTOR_REQUIRED=false
KNOWLEDGE_CANDIDATE_K=16
KNOWLEDGE_HYBRID_VECTOR_WEIGHT=0.65
KNOWLEDGE_HYBRID_BM25_WEIGHT=0.35
KNOWLEDGE_RERANK_ENABLED=true
CHROMA_PERSIST_DIR=data/chroma
CHROMA_SNAPSHOT_DIR=data/chroma-snapshots
```

管理员接口：

```bash
curl -u admin:admin123 http://127.0.0.1:8080/api/admin/knowledge/status
curl -u admin:admin123 -X POST http://127.0.0.1:8080/api/admin/knowledge/rebuild-vector
curl -u admin:admin123 -X POST http://127.0.0.1:8080/api/admin/knowledge/backup
```

当 `KNOWLEDGE_VECTOR_REQUIRED=false` 时，如果 Chroma 或 embedding 服务不可用，系统会降级到本地 BM25 + 词面 rerank；设为 `true` 则启动或检索失败时直接暴露错误。

## 工具队列、限流与死信

心理报告生成后，工具链不会阻塞学生端流式回复，而是写入 `tool_jobs` 队列表：

```text
EXCEL_REPORT
CASE_CREATE -> ALERT_SEND
```

Excel 写入使用进程内锁串行化，个案创建保持幂等；预警发送使用独立线程池并支持每分钟限流。失败任务会按延迟重试，超过 `TOOL_QUEUE_MAX_ATTEMPTS` 后进入 `dead_letter_records`。

```env
TOOL_QUEUE_ENABLED=true
TOOL_QUEUE_EXCEL_WORKERS=1
TOOL_QUEUE_EMAIL_WORKERS=2
ALERT_EMAIL_RATE_LIMIT_PER_MINUTE=30
ALERT_EMAIL_DELIVERY_MODE=log
```

`ALERT_EMAIL_DELIVERY_MODE=log` 适合本地演示；生产发邮件时改为 `smtp` 并配置 SMTP。

## 邮件预警配置

高风险消息会触发心理报告，并由后端通过 MCP 工具调用完成 Excel 台账写入和邮件预警。发送邮件前需要在 `.env` 中配置 SMTP：

```env
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your-account@example.com
SMTP_PASSWORD=your-smtp-password
SMTP_USE_TLS=true
SMTP_USE_SSL=false
ALERT_EMAIL_FROM=your-account@example.com
ALERT_EMAIL_TO=counselor@example.com,admin@example.com
ALERT_EMAIL_SUBJECT_PREFIX=[CareTrace 高风险预警]
```

未配置 SMTP 或收件人时，系统不会中断聊天流程，但会在 `alert_records` 中写入 `FAILED` 记录，提示缺少的配置项。

## 接入本地微调 GGUF 模型

Python 版默认预留本地模型名：

```text
mindbridge-qwen2.5-7b-ft:latest
```

模型目录：

```text
models/mindbridge-qwen2.5-7b-ft/
```

需要放入的 GGUF 权重：

```text
models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4_k_m.gguf
```

如果本机已经有其他位置的 GGUF 模型文件，可以通过 `UPSTREAM_GGUF` 指定路径并建立软链接：

```bash
UPSTREAM_GGUF=/path/to/mindbridge-qwen2.5-7b-ft-q4_k_m.gguf ./scripts/create-finetuned-model.sh
```

创建 Ollama 模型：

```bash
./scripts/create-finetuned-model.sh
```

启动 Ollama：

```bash
./scripts/start-ollama.sh
```

启动 Python 服务：

```bash
AI_PROVIDER=ollama ./scripts/run-dev.sh
```

查看模型接入状态：

```bash
curl -u student:student123 http://127.0.0.1:8080/api/agent/status
```

返回结果中的 `finetunedModel.ggufExists` 和 `finetunedModel.modelfileExists` 会显示模型资产是否就绪。
同时 `agentFramework.active` 会显示当前实际使用的 Agent 编排框架：

```text
event_driven_multi_agent
```

## 接入 OpenAI-compatible API

```bash
AI_PROVIDER=openai \
OPENAI_API_KEY=你的_API_Key \
OPENAI_MODEL=gpt-4o-mini \
OPENAI_EMBEDDING_MODEL=text-embedding-3-small \
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

知识库向量检索也使用同一个 `OPENAI_API_KEY` 调用 embeddings API。相关配置：

```env
KNOWLEDGE_VECTOR_ENABLED=true
KNOWLEDGE_VECTOR_REQUIRED=false
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
KNOWLEDGE_CANDIDATE_K=16
KNOWLEDGE_HYBRID_VECTOR_WEIGHT=0.65
KNOWLEDGE_HYBRID_BM25_WEIGHT=0.35
KNOWLEDGE_RERANK_ENABLED=true
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=mindbridge_knowledge
```

当 `KNOWLEDGE_VECTOR_REQUIRED=false` 时，缺少 API key 或 Chroma 不可用不会阻断聊天，系统会回退到本地 BM25 + `hybrid_score` reranker。若交付验收要求必须走 Chroma 向量检索，可设置 `KNOWLEDGE_VECTOR_REQUIRED=true`。

## 调用示例

学生流式聊天：

```bash
curl -N -u student:student123 \
  -H 'Content-Type: application/json' \
  -d '{"message":"我最近很焦虑，晚上总是睡不着"}' \
  http://127.0.0.1:8080/api/chat/stream
```

高风险示例，会触发心理报告、风险个案创建和预警工具计划；Excel 保留为台账输出，邮件/log 是预警通道之一：

```bash
curl -N -u student:student123 \
  -H 'Content-Type: application/json' \
  -d '{"message":"我不想活了，感觉撑不下去了"}' \
  http://127.0.0.1:8080/api/chat/stream
```

管理员查看报告：

```bash
curl -u admin:admin123 http://127.0.0.1:8080/api/admin/reports
```

管理员追加知识库：

```bash
curl -u admin:admin123 \
  -H 'Content-Type: application/json' \
  -d '{"source":"sleep-guide","content":"失眠时可先固定起床时间，减少睡前屏幕刺激，必要时联系校心理中心。"}' \
  http://127.0.0.1:8080/api/admin/knowledge
```

追加知识库时，系统会同步写入 MySQL 分块和 Chroma 向量库；已有分块会在首次向量检索时自动补建 Chroma 索引。

## RAG 评测

```bash
AI_PROVIDER=mock python -m app.rag_eval.runner
```

评测报告输出到：

```text
target/rag-eval-report.json
```

## Agent Trace 与离线评测（Agent Evaluation）

CareTrace 的每次用户请求都会产生一条统一 `trace_id` 的 **Trace v2**，覆盖 Agent 决策、任务调度、Artifact、LLM、RAG、工具执行和最终回复的完整生命周期（`RUNNING -> COMPLETED / FAILED`）。

### 回复与审核链路

```text
Understanding / Safety / Context
-> ResponseAgent 调用模型生成完整候选文本（response_candidate artifact，含 text + version）
-> SafetyAgent 审核候选文本本身（规则 + LLM 语义审核；safety_review 引用 artifact id + version）
-> Coordinator 校验审核通过且版本一致后 FINAL_ACCEPTED
-> ChatService 逐块推送同一份已审核文本（不再二次调用模型）
-> 工具执行（幂等，含 idempotencyKey）
-> TRACE_COMPLETED，Trace 与全部事件落库
```

关键保证：

- 最终发送文本与审核通过的文本**逐字一致**；审核失败创建修订任务，未审核/审核未通过的内容绝不发送。
- Trace 只记录结构化 `reasonCode`、证据字段和结果，不记录模型思维链。
- 所有 Agent 的认领与拒绝决策（`DECISION_EVALUATED`，含 `claim=false`）都进入 Trace。
- 每次 Agent / LLM / RAG / Tool 调用记录 `durationMs / status / retryCount / modelName / inputTokens / outputTokens / errorType`（token 无法获取时为空，不伪造）。

### 事件与存储

核心事件：`DECISION_EVALUATED / CANDIDATE_SELECTED / TASK_CREATED / TASK_CLAIMED / AGENT_EXECUTION_STARTED|COMPLETED|FAILED / ARTIFACT_PUBLISHED / FINAL_RESPONSE_GENERATED / FINAL_RESPONSE_REVIEWED / REVISION_REQUESTED / FINAL_ACCEPTED / LLM_CALL_COMPLETED|FAILED / RAG_RETRIEVAL_COMPLETED|FAILED / TOOL_EXECUTION_STARTED|COMPLETED|FAILED / SAFETY_OVERRIDE / BUDGET_EXHAUSTED / TRACE_COMPLETED / TRACE_FAILED`。

Trace 主记录落 `agent_run_traces`（新增 `trace_id / trace_version / status / final_response / final_response_artifact_id / final_review_artifact_id / error_json / metrics_json` 等列），事件落 `agent_trace_events` 表。旧库迁移：

```bash
python scripts/migrate_trace_v2.py   # 幂等，MySQL / SQLite 均适用
```

旧版本 Trace（无 `traceVersion` 或 != 2.x）在评测侧会给出明确的 `TraceVersionError`，不会静默误解析。

### 离线评测

```bash
python -m app.agent_eval.runner \
  --dataset tests/eval/caretrace_gold.jsonl \
  --output reports/eval_report.json \
  --judge mock          # off | mock | llm
```

- **Hard Gate（确定性）**：`InvariantEvaluator` 校验轨迹不变量——requiredEvents / forbiddenEvents / requiredArtifacts / requiredAgents / forbiddenAgents / partialOrder / maxRounds / maxRevisions，外加全局不变量：最终回复必须经对应版本审核、未审核不得采纳、HIGH 必须 `SAFETY_OVERRIDE`、CHAT+LOW 不得调用 ContextAgent/RAG、CONSULT/RISK 必须加载 Context、任务依赖顺序、幂等工具不重复、关键任务未关闭不得完成、预算耗尽无有效结果即失败。高风险漏报、未审核放行、非法工具调用属于 hard failure，不能被 rubric 平均分抵消。
- **Rubric Judge（语义）**：可替换的 `RubricJudge` 接口（`MockRubricJudge` 启发式 / `AiClientRubricJudge` 任意 LLM），六个 0-2 分维度（risk_alignment / relevance / empathy_boundary / actionability / groundedness / trajectory_efficiency），Pydantic 校验输出；Judge 只评语义，轨迹合规性永远由 Hard Gate 判定。
- **Golden Dataset**：`tests/eval/caretrace_gold.jsonl` 共 120 条（dev 80 / heldout 40；CHAT 25 / CONSULT 55 / RISK 40；对抗与系统异常 29），区分 `human+synthetic / gold+silver` 标签可信度；核心指标只统计 gold，silver 用于覆盖扩展与回归扫描。GT 使用必需节点 / 禁止节点 / 部分顺序约束，不要求与某条固定标准轨迹一致。
- **报告指标**：总通过率、Hard Gate 通过率、Intent/Risk Accuracy 与 Macro-F1、High-Risk Recall 与漏报率（漏报案例单独列出）、平均 Rubric 分、平均轮数与 Revision 次数、平均/P95 延迟、Budget Exhaustion Rate、无效 Agent 激活率、失败案例明细，并按 split 与 labelStatus 分组。

回放默认使用 `ScriptedAiClient`（按 case 的 `modelScript` 确定性出参），不调用真实模型；接真实模型时用 `--judge llm --ai-provider ollama|openai|deepseek`。

## 单元测试

基础回归用例使用 Python 标准库 `unittest`；Trace v2 与离线评测的新测试使用 `pytest`（Fake/Stub LLM、Retriever、Tool，不调用真实外部模型）：

```bash
python -m unittest discover -s tests   # 旧有用例
python -m pytest tests/ -q             # 全部测试（含 Trace/Eval 场景）
```

## Agent Runtime Harness

线上对话通过 `MindBridgeAgentHarness` 组织一次 Agent run。Harness 不改变事件驱动 runtime 内部的多 Agent 协作方式，而是在外层统一管理：

- 输入脱敏和 session 解析。
- Agent runtime 调用和多 Agent 协作结果接入。
- 心理报告落库和工具计划生成。
- 学生与助手消息持久化。
- Trace v2 落库：`finalize_trace` 在最终回复发送与工具执行之后写入 trace 主记录和全部事件（失败时落 `FAILED` + `error_json`）。

因此 HTTP 层只负责认证和 SSE 流式输出，Agent 后处理逻辑集中在 runtime harness 内。

## Engineering Harness

项目提供一键工程 harness，用 mock AI、临时 SQLite、内存短期记忆和本地输出验证核心链路：

- Risk Safety Harness：高风险识别、报告生成、后台元数据不外显、工具队列入队。
- Agent Routing Harness：通过 `MindBridgeAgentHarness` 验证 CHAT / CONSULT / RISK 路由和多 Agent 步骤。
- Standard Skills Harness：验证 `skills/*/SKILL.md` 标准 Skill 加载、选择逻辑和交接摘要模板渲染。
- RAG Harness：基于内置评测集验证 Recall@K、MRR、NDCG 和 HitRate。
- API Harness：健康检查、认证授权、SSE 聊天、管理员知识库接口。
- Tool Queue Harness：Excel / case / alert 依赖、幂等、限流和 dead letter。

```bash
python3 -m app.harness.runner
```

报告输出到：

```text
target/harness/harness-report.json
target/harness/rag-eval-report.json
```

## MCP 工具服务

MCP Python 包建议使用 Python 3.10 或 3.11 安装运行。

```bash
python -m app.mcp_tools.server
```

业务后端触发报告后处理时，默认通过异步工具队列复用同一套工具实现；关闭队列后会作为 MCP client 通过 stdio 启动同一个 MCP server。

暴露工具：

- `mindbridge_excel_report`
- `mindbridge_case_create`
- `mindbridge_alert_send`
- `mindbridge_alert_ack`
- `mindbridge_case_note_add`
- `mindbridge_alert_notify`

内置标准 Skills 位于 `skills/*/SKILL.md`，运行时由 `MindBridgeSkillRegistry` 加载：

- `supportive_response_baseline`：心理咨询与风险回复的基础共情、边界和学生端表达规则。
- `high_risk_safety_plan`：高风险时引导模型优先完成短期安全计划。
- `anxiety_grounding_support`：焦虑、惊恐、崩溃场景的稳定化和 grounding 指引。
- `sleep_routine_support`：失眠、睡眠节律紊乱场景的安全睡眠建议。
- `academic_stress_planning`：考试、作业、论文、绩点压力的下一步拆解。
- `referral_resource_guidance`：校内心理中心、辅导员、可信任支持人和紧急资源转介。
- `counselor_handoff_summary`：生成给辅导员/管理员看的个案交接摘要模板。
