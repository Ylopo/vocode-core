from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from vocode.streaming.agent.token_utils import (
    get_chat_gpt_max_tokens,
    num_tokens_from_functions,
    num_tokens_from_messages,
)
from vocode.streaming.models.actions import FunctionFragment, PhraseBasedActionTrigger
from vocode.streaming.models.agent import LLM_AGENT_DEFAULT_MAX_TOKENS
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    ConferenceEvent,
    EventLog,
    Message,
    Transcript,
)

def is_tools_format_model(model_name: str) -> bool:
    """
    Returns True for models that require the tools/tool format (gpt-5+).
    Returns False for models that use the legacy functions/function_call format (gpt-4 and below).
    """
    tools_format_prefixes = ("gpt-5",)
    return any(model_name.startswith(prefix) for prefix in tools_format_prefixes)

def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}


def is_phrase_based_action_event_log(event_log: EventLog) -> bool:
    return (
        (isinstance(event_log, ActionStart) or isinstance(event_log, ActionFinish))
        and event_log.action_input is not None
        and event_log.action_input.action_config is not None
        and isinstance(
            event_log.action_input.action_config.action_trigger, PhraseBasedActionTrigger
        )
    )


def get_openai_chat_messages_from_transcript(
    merged_event_logs: List[EventLog],
    prompt_preamble: str,
    model_name: str,
) -> List[dict]:
    use_tools_format = is_tools_format_model(model_name)
    chat_messages = [{"role": "system", "content": prompt_preamble}]
    for event_log in merged_event_logs:
        if isinstance(event_log, Message):
            if len(event_log.text.strip()) == 0:
                continue
            else:
                chat_messages.append(
                    {
                        "role": ("assistant" if event_log.sender == Sender.BOT else "user"),
                        "content": event_log.to_string(include_sender=False),
                    },
                )
        elif isinstance(event_log, ActionStart):
            action_message: Dict[str, Any]
            if is_phrase_based_action_event_log(event_log=event_log):
                pass
            else:
                if use_tools_format:
                    action_message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{event_log.action_type}",
                                "type": "function",
                                "function": {
                                    "name": event_log.action_type,
                                    "arguments": event_log.action_input.params.json(),
                                },
                            }
                        ],
                    }
                else:
                    action_message = {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": event_log.action_type,
                            "arguments": event_log.action_input.params.json(),
                        },
                    }
                chat_messages.append(action_message)
        elif isinstance(event_log, ActionFinish):
            if is_phrase_based_action_event_log(event_log=event_log):
                pass
            else: 
                if use_tools_format:
                    action_message = {
                        "role": "tool",
                        "tool_call_id": f"call_{event_log.action_type}",
                        "content": event_log.to_string(include_header=False),
                    }
                else:
                    action_message = {
                        "role": "function",
                        "name": event_log.action_type,
                        "content": event_log.to_string(include_header=False),
                    }
                chat_messages.append(action_message)
        elif isinstance(event_log, ConferenceEvent):
            chat_messages.append(
                {"role": "user", "content": event_log.to_string(include_sender=False)},
            )
    return chat_messages


def merge_event_logs(event_logs: List[EventLog]) -> List[EventLog]:
    """Returns a new list of event logs where consecutive bot messages are merged."""
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(event_log.text for event_log in bot_messages_buffer)
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    return new_event_logs


def format_openai_chat_messages_from_transcript(
    transcript: Transcript,
    model_name: str,
    functions: Optional[List[Dict]],
    prompt_preamble: str,
) -> List[dict]:
    # merge consecutive bot messages
    merged_event_logs: List[EventLog] = merge_event_logs(event_logs=transcript.event_logs)

    chat_messages: List[Dict[str, Optional[Any]]]
    chat_messages = get_openai_chat_messages_from_transcript(
        merged_event_logs=merged_event_logs,
        prompt_preamble=prompt_preamble,
        model_name=model_name,
    )

    context_size = num_tokens_from_messages(
        messages=chat_messages,
        model=model_name,
    ) + num_tokens_from_functions(functions=functions, model=model_name)

    num_removed_messages = 0
    while (
        context_size > get_chat_gpt_max_tokens(model_name) - LLM_AGENT_DEFAULT_MAX_TOKENS - 50
    ):  # context limit includes the max tokens, and 50 for safety
        if len(chat_messages) <= 1:
            logger.error(f"Prompt is too long to fit in context window, num tokens {context_size}")
            break
        num_removed_messages += 1
        chat_messages.pop(1)
        # Remove any orphaned tool/function responses that lost their preceding
        # tool_calls message after the pop above, which would cause a 400 error.
        while len(chat_messages) > 1 and chat_messages[1].get("role") in ("tool", "function"):
            num_removed_messages += 1
            chat_messages.pop(1)
        context_size = num_tokens_from_messages(
            messages=chat_messages,
            model=model_name,
        ) + num_tokens_from_functions(functions=functions, model=model_name)

    if num_removed_messages > 0:
        logger.info(
            "Removed %d messages from prompt to satisfy context limit",
            num_removed_messages,
        )

    return chat_messages


async def openai_get_tokens(
    gen: AsyncGenerator[ChatCompletionChunk, None],
) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    async for event in gen:
        choices = event.choices
        if len(choices) == 0:
            continue
        choice = choices[0]
        if choice.finish_reason:
            if choice.finish_reason == "content_filter":
                logger.warning(
                    "Detected content filter.",
                    extra={"chat_completion_chunk": event.model_dump()},
                )
            break
        delta = choice.delta
        if delta.content is not None:
            token = delta.content
            yield token
        elif delta.function_call is not None:
            yield FunctionFragment(
                name=(delta.function_call.name if delta.function_call.name is not None else ""),
                arguments=(
                    delta.function_call.arguments
                    if delta.function_call.arguments is not None
                    else ""
                ),
            )
        elif delta.tool_calls is not None:
            for tool_call in delta.tool_calls:
                if tool_call.function is not None:
                    yield FunctionFragment(
                        name=(tool_call.function.name or ""),
                        arguments=(tool_call.function.arguments or ""),
                    )
                    break
