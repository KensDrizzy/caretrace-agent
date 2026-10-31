# 三层学生记忆与增量上下文压缩

移植依据：WeaveMind `core/memory.py` 的持久去重、向量/bigram 双路召回和时间/频次重排，以及 `core/compaction.py` 的分片摘要、合并与压缩前事实沉淀。按 CareTrace 的多学生场景改用 SQL 持久化和服务端用户隔离。

## 三层结构

- 工作记忆：MySQL 会话原文为持久来源，Redis 缓存/降级；模型只接收最近消息，单条默认最多 600 字。当前输入由 runtime 单独加入一次。
- 核心记忆：学生明确表达的回复偏好，默认最近 3 条常驻提示上下文，例如“我更喜欢简短的建议”。
- 长期事实：学生明确表达的持续处境，例如“我正在准备考研”。按当前问题召回默认 5 条，带来源消息及更新时间。只接受学生消息中的原文片段，助手回复、网页内容和推断诊断不能进入事实库。

`student_memories` 保存用户、来源会话/消息、类型、内容、SHA-256 摘要、向量/模型指纹、时间及访问统计。`(user_id,digest)` 唯一约束和数据库 savepoint 保证重复写入不增加副本。事实写入与用户消息在同一事务内提交；embedding 失败不阻断事实落库。

不同用户的写入、查询和访问统计均在数据库层按 user_id 限制。软删除会话后，其来源事实不再召回。原始会话与报告仍按原有规则保留。新表在后端启动时由 `create_schema()` 创建；已有会话表通过兼容迁移补充 `deleted_at` 字段。

## 双路召回及重排

1. 按用户读取有界候选，默认最多 200 条近期长期事实；核心偏好另取，避免被大量事实挤掉。
2. 字符 bigram Jaccard + 子串命中计算词法分；同一 embedding 模型空间的余弦相似度计算语义分。默认准入门槛分别为 0.08、0.60。
3. 两路至少一路达到门槛，长期事实才可进入重排；频次本身不能让无关事实被召回。核心偏好单独常驻。
4. `max(lexical, semantic) * (0.3 + 0.7 * 0.5^(age_days/7)) * importance + min(0.3, 0.05*log1p(access_count))`。
5. 命中时原子递增访问次数并更新 last_access；embedding 模型/端点指纹变化时补齐向量，失败则退回词法检索，不把哈希向量当作真实语义能力。

复用已有 `EMBEDDING_*` 配置及其 OpenAI-compatible `/embeddings` 服务。未配置凭证或服务不可用时保留 bigram 路。小规模事实向量保存在 SQL 中，在有界候选集上计算余弦；没有为记忆额外启动向量数据库。它不是全量 ANN 检索，大量事实时需要升级候选召回。

## 增量 Map-Reduce

`conversation_summaries` 按会话保存 summary 和已处理消息 ID 游标。读取数据库中游标之后、近期窗口之前的消息，每批最多 64 条；不受 Redis 最近 40 条窗口限制。

- Map：按默认 1200 字分片，从学生消息抽取有界原话要点；助手内容不作为学生事实。
- Reduce：合并已有摘要和分片要点，去重并按近期优先压入摘要预算，默认 500 字。
- 丢弃旧原文前先尝试沉淀合格事实；单轮聊天中的合格事实也会随消息立即持久化。
- 通过行锁和游标实现增量处理，避免每次把全部旧对话重新压缩。
- 输出核心/召回记忆与摘要的有效 JSON 数据，默认最多 1800 字，配合近期消息构造模型上下文；明确标记为可能过时的数据，不授权执行其中指令或据此判断当前风险。

当前默认实现是**确定性的抽取式 Map-Reduce**，不额外调用 LLM 做抽象摘要。优点是没有额外摘要模型费用和事实幻觉；代价是表达质量弱于语义摘要，也会丢失未被选中的细节。近期窗口和持久事实是补充保留渠道，不保证所有历史信息无损。

## 接入点与观测

- `app/services/student_memory.py`：模型、事实提取、双路检索和压缩。
- `app/agents/event_driven_runtime.py`：每轮准备记忆，短事务提交后再进入模型调用；失败回退旧压缩流程。
- `app/agents/harness.py`：保存学生消息时沉淀事实。
- `app/agents/autonomous.py`：ContextAgent 沿用准备好的记忆，ResponseAgent 使用同一份模型历史。
- Trace `MEMORY_PREPARED`：written、recalled、compressedMessages、vectorAvailable；准备失败时 degraded=true。

`STUDENT_MEMORY_ENABLED=false` 可以回到原有记忆流程；`MEMORY_SEMANTIC_ENABLED=false` 只关闭向量路。其他预算配置见 `.env.example`。

例子：学生 A 在旧会话说“我正在准备考研”“我更喜欢简短的建议”；在新会话问“硕士入学的复习怎么安排”，可由向量路召回考研事实，核心偏好常驻。学生 B 的请求不读取 A 的事实。此场景已使用隔离 SQLite、模拟 embedding 和真实 Agent runtime 验证；线上实际语义质量取决于 embedding 模型。

验证：`.venv/bin/python -m pytest -q tests/test_student_memory.py tests/test_memory_compaction.py`。


## 完整轮次和单次注入

近期消息数量为软目标：若窗口从助手回复开始，扩展到该轮用户消息，摘要游标也停在该轮之前。最终回答中摘要只注入一次；ContextAgent 的检索可以继续使用同一 brief。超长消息仍受字符配置约束，此版本不宣称精确 token 预算。

## 跨层去重与最终输入预算

准备上下文时，只把近期学生原文和当前输入作为去重依据：已出现的逐字事实不再从长期库重复注入；摘要中已由原文或召回事实覆盖的要点也不再重复。去重只影响本次 Prompt，不删除持久记忆。不使用语义相似度合并事实，避免把否定、修正或时间变化误判为重复。注入的事实包含来源会话 ID、消息 ID 和更新时间，便于核查。

`RESPONSE_PROMPT_MAX_CHARS` 默认 18000，作用于 ResponseAgent 组装后的实际消息内容（含系统规则、Skill、知识、记忆、网页证据和历史）。超限时按顺序移除最旧的完整对话轮次、可选上下文消息，保留前两条规则/Skill消息及当前用户输入。若必须保留的内容自身超限，完整保留并在响应 artifact 的 `contextBudget.exceeded` 中标记；不会静默截断用户请求。知识证据另有原有的单独上限，规则/Skill与用户输入构成软预算下界。

这是**字符软预算**，不等同于不同模型的 token 窗口。`contextBudget` 记录 beforeChars、afterChars、removedTurns、removedContextMessages 与 exceeded；记忆准备事件记录 injectedMemories、memoryChars。模型的输出 token 配置独立生效。
