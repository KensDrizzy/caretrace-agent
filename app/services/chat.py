from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy.orm import Session

from app.agents.harness import MindBridgeAgentHarness
from app.core.config import Settings
from app.models.entities import UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient, ModelConnectionError, ModelNotFoundError, ModelTimeoutError, split_text


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.agent_harness = MindBridgeAgentHarness(db, settings)

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        # The agent runtime is synchronous and may perform blocking LLM/DB calls.
        # Run it in the default thread pool so the ASGI event loop stays responsive.
        try:
            outcome = await asyncio.to_thread(self.agent_harness.run, user, request)
        except ModelNotFoundError:
            yield sse("error", ChatStreamEvent(type="error", sessionId=request.sessionId, message="模型未找到，请检查模型是否已加载或运行 create-finetuned-model.sh").model_dump(by_alias=True))
            return
        except (ModelConnectionError, ModelTimeoutError):
            yield sse("error", ChatStreamEvent(type="error", sessionId=request.sessionId, message="模型连接失败，服务正在启动或不可用，请稍后再试").model_dump(by_alias=True))
            return
        except Exception:
            logger.exception("Agent harness failed for session=%s", request.sessionId)
            yield sse("error", ChatStreamEvent(type="error", sessionId=request.sessionId, message="处理失败，请稍后再试").model_dump(by_alias=True))
            return

        yield sse("meta", ChatStreamEvent(type="meta", sessionId=outcome.session.public_id).model_dump(by_alias=True))
        try:
            # 直接发送 ResponseAgent 生成、SafetyAgent 审核过的同一份文本，不再调用模型
            for chunk in split_text(outcome.final_response, 12):
                yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=chunk).model_dump())
            if outcome.final_response:
                self.agent_harness.save_assistant_message(user, outcome.session, outcome.final_response)
            tool_records = await self.agent_harness.dispatch_tools(outcome.tool_plan)
        except Exception as exc:
            logger.exception("Post-harness stage failed for session=%s", outcome.session.public_id)
            self.agent_harness.finalize_trace(outcome, [], error=exc)
            yield sse("error", ChatStreamEvent(type="error", sessionId=outcome.session.public_id, message="处理失败，请稍后再试").model_dump())
            return
        self.agent_harness.finalize_trace(outcome, tool_records)
        yield sse("done", ChatStreamEvent(type="done", sessionId=outcome.session.public_id).model_dump())


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
