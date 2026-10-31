from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    agent_framework: str = "event_driven_multi_agent"
    agent_max_rounds: int = 8
    agent_max_claims_per_round: int = 4
    agent_max_claims_per_agent: int = 3
    agent_final_acceptance_min_confidence: float = 0.6
    agent_model_default_provider: str = ""
    agent_model_default_model: str = ""
    agent_model_coordinator_provider: str = ""
    agent_model_coordinator_model: str = ""
    agent_model_coordinator_max_tokens: int = 256
    agent_model_understanding_provider: str = ""
    agent_model_understanding_model: str = ""
    agent_model_understanding_max_tokens: int = 256
    agent_model_safety_provider: str = ""
    agent_model_safety_model: str = ""
    agent_model_safety_max_tokens: int = 128
    # SafetyAgent 内部 per-task 拆分：情绪/风险分类走 safety（可指向本地微调模型），
    # 候选回复的语义审核走 safety_review（需要强推理，通常指向 API 大模型）。
    agent_model_safety_review_provider: str = ""
    agent_model_safety_review_model: str = ""
    agent_model_safety_review_max_tokens: int = 256
    agent_model_context_provider: str = ""
    agent_model_context_model: str = ""
    agent_model_context_max_tokens: int = 256
    agent_model_response_provider: str = ""
    agent_model_response_model: str = ""
    agent_model_response_max_tokens: int = 768
    ai_provider: str = "ollama"
    ai_temperature: float = 0.35
    ai_max_tokens: int = 2048
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mindbridge-qwen2.5-7b-ft:latest"
    ollama_think: bool = False
    ollama_timeout_seconds: float = 180.0
    # Qwen3 campus prompts include compact RAG evidence and agent instructions.
    # Keep enough headroom for a full support turn instead of Ollama's 4096 default.
    ollama_num_ctx: int = 4096
    finetuned_model_name: str = "mindbridge-qwen2.5-7b-ft:latest"
    finetuned_model_dir: str = "finetuning/qwen25-7b-lora-mood-classify"
    finetuned_model_file: str = "mindbridge-qwen2.5-7b-ft-q4_k_m.gguf"
    # Qwen3-14B 校园心理场景模型：用于理解、检索改写、回复生成和安全语义审核。
    campus_model_name: str = "caretrace-qwen3-14b-ft:latest"
    campus_model_dir: str = "finetuning/qwen3-14b-lora-full"
    campus_model_file: str = "caretrace-qwen3-14b-ft-q4_k_m.gguf"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str = "none"
    openai_embedding_model: str = "text-embedding-3-small"
    # Embedding 可独立于对话模型配置；为空时兼容旧的 OPENAI_* 配置。
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    database_url: str = "mysql+pymysql://mindbridge:mindbridge@127.0.0.1:3306/mindbridge?charset=utf8mb4"
    chat_history_limit: int = 10
    # 已审核文本重放的打字机节奏（毫秒/块，0 表示不节流）
    chat_stream_chunk_delay_ms: float = 25.0
    knowledge_top_k: int = 4
    knowledge_candidate_k: int = 16
    knowledge_chunk_size: int = 512
    knowledge_chunk_overlap: int = 64
    knowledge_parent_chunk_size: int = 2000
    # Keep the response prompt within the default 4K context window even when
    # the knowledge store returns a long multi-question guide.
    response_context_max_chars: int = 3200
    response_prompt_max_chars: int = Field(default=18000, ge=1000, le=200000)
    knowledge_hybrid_vector_weight: float = 0.65
    knowledge_hybrid_bm25_weight: float = 0.35
    knowledge_rerank_enabled: bool = True
    knowledge_vector_enabled: bool = True
    knowledge_vector_required: bool = False
    knowledge_bm25_max_docs: int = 500
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "mindbridge_knowledge"
    chroma_snapshot_dir: str = "data/chroma-snapshots"
    chroma_snapshot_keep: int = 5
    embedding_timeout_seconds: float = 30.0
    rag_eval_dataset: str = "app/rag_eval/mindbridge-rag-eval.json"
    rag_eval_output: str = "target/rag-eval-report.json"
    rag_eval_enabled: bool = False
    rag_eval_exit_after_run: bool = False
    agentic_rag_enabled: bool = True
    agentic_rag_max_iterations: int = 2
    agentic_rag_sufficiency_threshold: float = 0.7
    agentic_rag_eval_dataset: str = "app/rag_eval/mindbridge-agentic-rag-eval.json"
    agentic_rag_eval_output: str = "target/agentic-rag-eval-report.json"
    excel_path: str = "data/caretrace-risk-ledger.xlsx"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_memory_ttl_seconds: int = 86400
    redis_memory_max_messages: int = 40
    redis_socket_timeout_seconds: float = 2.0
    memory_compaction_enabled: bool = True
    memory_compaction_recent_messages: int = 8
    memory_summary_max_chars: int = 500
    student_memory_enabled: bool = True
    memory_semantic_enabled: bool = True
    memory_candidate_limit: int = Field(default=200, ge=1, le=2000)
    memory_core_limit: int = Field(default=3, ge=0, le=10)
    memory_recall_limit: int = Field(default=5, ge=0, le=20)
    memory_half_life_days: float = Field(default=7, gt=0)
    memory_lexical_min_score: float = Field(default=0.08, gt=0, le=1)
    memory_semantic_min_score: float = Field(default=0.60, gt=0, le=1)
    memory_embedding_timeout_seconds: float = Field(default=5, gt=0, le=30)
    memory_map_chunk_chars: int = Field(default=1200, ge=200, le=4000)
    memory_prompt_max_chars: int = Field(default=1800, ge=300, le=4000)
    memory_message_max_chars: int = Field(default=600, ge=100, le=2000)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: float = 10.0
    alert_email_delivery_mode: str = "log"
    alert_email_from: str = ""
    alert_email_to: str = ""
    alert_email_subject_prefix: str = "[CareTrace 高风险预警]"
    tool_queue_enabled: bool = True
    tool_queue_poll_interval_seconds: float = 1.0
    tool_queue_batch_size: int = 10
    tool_queue_max_attempts: int = 3
    tool_queue_retry_delay_seconds: float = 15.0
    tool_queue_excel_workers: int = 1
    tool_queue_email_workers: int = 2
    alert_email_rate_limit_per_minute: int = 30

    # 本机演示：只能由显式绑定的学生账号使用宿主机 Chrome 登录态。
    web_access_enabled: bool = False
    web_access_username: str = ""
    web_access_browser_url: str = ""  # 空值使用 Chrome autoConnect
    web_access_mcp_command: str = "npx"
    web_access_mcp_args: list[str] = ["--yes", "chrome-devtools-mcp@1.12.0"]
    web_access_node_bin: str = ""  # 可指定 Node 22 的 bin 目录
    web_access_timeout_seconds: float = Field(default=90, ge=10, le=180)
    web_access_tool_timeout_seconds: float = Field(default=20, ge=1, le=60)
    web_access_max_results: int = Field(default=3, ge=1, le=5)
    web_access_persistent: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_embedding_base_url(self) -> str:
        return (self.embedding_base_url.strip() or self.openai_base_url.strip()).rstrip("/")

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key.strip() or self.openai_api_key.strip()

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model.strip() or self.openai_embedding_model.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
