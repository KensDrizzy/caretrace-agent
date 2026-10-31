from app.schemas.dtos import AiMessage
from app.services.prompt_budget import fit_prompt


def msg(role, content):
    return AiMessage(role=role, content=content)


def test_trims_complete_old_turns_and_keeps_rules_current_input():
    rules = [msg('system', 'risk-rules'), msg('system', 'selected-skills')]
    context = msg('system', 'facts' * 10)
    history = [msg('user', 'old-question' * 10), msg('assistant', 'old-reply' * 10),
        msg('assistant', 'old-follow-up' * 10)]
    current = msg('user', 'new-question')
    result = fit_prompt([*rules, context, *history, current], 100)
    assert result.messages == [*rules, context, current]
    assert result.stats['removedTurns'] == 1
    assert result.stats['afterChars'] <= 100
    assert not result.stats['exceeded']


def test_context_is_removed_as_whole_data_message():
    messages = [msg('system', 'rules'), msg('system', 'skills'),
        msg('system', '{"facts": "' + 'x' * 100 + '"}'), msg('user', 'question')]
    result = fit_prompt(messages, 30)
    assert result.messages == [messages[0], messages[1], messages[-1]]
    assert result.stats['removedContextMessages'] == 1


def test_oversized_current_input_is_preserved_and_overflow_observable():
    messages = [msg('system', 'rules'), msg('system', 'skills'), msg('user', 'x' * 1000)]
    result = fit_prompt(messages, 100)
    assert result.messages == messages
    assert result.stats['exceeded']
    assert result.stats['afterChars'] == 1011
