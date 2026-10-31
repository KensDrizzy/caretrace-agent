"""Character-based prompt admission; never truncate rules or the current request."""
from __future__ import annotations

from dataclasses import dataclass
from app.schemas.dtos import AiMessage


@dataclass(frozen=True)
class PromptBudgetResult:
    messages: list[AiMessage]
    stats: dict


def fit_prompt(messages: list[AiMessage], max_chars: int, protected_prefix: int = 2) -> PromptBudgetResult:
    """Remove oldest complete turns, then optional context messages.

    The initial rule/Skill messages and final user turn are mandatory. If these
    alone exceed the soft budget, report overflow instead of hiding user input.
    This counts actual Unicode characters, NOT model-specific tokens.
    """
    before = sum(len(m.content) for m in messages)
    if not messages:
        return PromptBudgetResult([], dict(beforeChars=0, afterChars=0, removedTurns=0,
            removedContextMessages=0, maxChars=max_chars, exceeded=False))
    prefix_size = min(protected_prefix, len(messages))
    fixed = messages[:prefix_size]
    remaining = messages[prefix_size:]
    first_user = next((i for i, m in enumerate(remaining) if m.role == 'user'), len(remaining))
    optional = remaining[:first_user]
    turns: list[list[AiMessage]] = []
    for message in remaining[first_user:]:
        if message.role == 'user':
            turns.append([])
        turns[-1].append(message)
    removed_turns = removed_context = 0
    size = before
    while size > max_chars and len(turns) > 1:
        size -= sum(len(m.content) for m in turns.pop(0))
        removed_turns += 1
    while size > max_chars and optional:
        size -= len(optional.pop().content)
        removed_context += 1
    result = [*fixed, *optional, *(m for turn in turns for m in turn)]
    return PromptBudgetResult(result, dict(beforeChars=before, afterChars=size,
        removedTurns=removed_turns, removedContextMessages=removed_context,
        maxChars=max_chars, exceeded=size > max_chars))
