from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

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
) -> List[dict]:
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


def _to_fc_id(call_id: str) -> str:
    """Responses API requires function call IDs to start with 'fc_', not 'call_'."""
    if call_id.startswith("call_"):
        return "fc_" + call_id[5:]
    if not call_id.startswith("fc_"):
        return "fc_" + call_id
    return call_id


def format_openai_responses_input_from_transcript(
    chat_messages: List[Dict],
) -> Tuple[Optional[str], List[Dict]]:
    """
    Convert a chat-completions message list into the (instructions, input) pair
    expected by the OpenAI Responses API.

    - System messages become the `instructions` string.
    - Assistant tool_calls / function_call entries become top-level function_call items.
    - tool / function result entries become top-level function_call_output items.
    - IDs are normalised from the 'call_' prefix to the 'fc_' prefix the Responses API requires.
    - Transcript-derived IDs get a per-invocation suffix. The transcript only records the
      action *type*, so calling one action twice would otherwise emit duplicate call_ids and
      the Responses API could not pair each call with its own output.
    """
    instructions: Optional[str] = None
    input_messages: List[Dict] = []

    # Calls are queued per action name and dequeued FIFO by their matching output, so the
    # nth result is always paired with the nth call of that name.
    call_counts: Dict[str, int] = {}
    pending_call_ids: Dict[str, List[str]] = {}

    def next_call_id(name: str) -> str:
        call_counts[name] = call_counts.get(name, 0) + 1
        fc_id = _to_fc_id(f"call_{name}_{call_counts[name]}")
        pending_call_ids.setdefault(name, []).append(fc_id)
        return fc_id

    def take_call_id(name: str) -> Optional[str]:
        queue = pending_call_ids.get(name)
        if queue:
            return queue.pop(0)
        return None

    for msg in chat_messages:
        role = msg.get("role")

        if role == "system":
            instructions = msg.get("content", "")
            continue

        if role in ("user", "assistant"):
            tool_calls = msg.get("tool_calls")
            function_call = msg.get("function_call")

            if tool_calls:
                # Any text content comes first as a plain assistant message
                if msg.get("content"):
                    input_messages.append({"role": "assistant", "content": msg["content"]})
                # Each function call is a top-level item - NOT wrapped in a role/content array
                for tc in tool_calls:
                    fc_id = _to_fc_id(tc["id"])
                    input_messages.append(
                        {
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": fc_id,
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    )
            elif function_call:
                fc_id = next_call_id(function_call["name"])
                if msg.get("content"):
                    input_messages.append({"role": "assistant", "content": msg["content"]})
                input_messages.append(
                    {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": fc_id,
                        "name": function_call["name"],
                        "arguments": function_call["arguments"],
                    }
                )
            else:
                input_messages.append({"role": role, "content": msg.get("content", "")})

        elif role == "tool":
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": _to_fc_id(msg["tool_call_id"]),
                    "output": msg.get("content", ""),
                }
            )

        elif role == "function":
            name = msg["name"]
            call_id = take_call_id(name)
            if call_id is None:
                # The matching call is not in this window - format_openai_chat_messages_from
                # _transcript trims from the front, so an ActionStart can be dropped while its
                # ActionFinish survives. A function_call_output with no call would be rejected
                # outright, so preserve the result as plain text instead of losing it.
                logger.debug(
                    f"No function_call in context for {name!r} result; "
                    "passing it through as a user message"
                )
                input_messages.append(
                    {
                        "role": "user",
                        "content": f"[Result of {name}]: {msg.get('content', '')}",
                    }
                )
            else:
                input_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": msg.get("content", ""),
                    }
                )

    return instructions, input_messages


def _incomplete_reason(event: Any) -> str:
    """Why the model stopped early - 'max_output_tokens', 'content_filter', ..."""
    details = getattr(getattr(event, "response", None), "incomplete_details", None)
    return getattr(details, "reason", None) or "unknown"


def _response_error(event: Any) -> str:
    error = getattr(getattr(event, "response", None), "error", None)
    if error is None:
        return "unknown error"
    return f"{getattr(error, 'code', None)}: {getattr(error, 'message', None)}"


async def responses_get_tokens(
    gen: AsyncGenerator,
) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    """Extract text tokens and function-call fragments from a Responses API stream."""
    current_function_name = ""
    async for event in gen:
        event_type = getattr(event, "type", None)

        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                current_function_name = getattr(item, "name", "") or ""

        elif event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                yield delta

        elif event_type == "response.function_call_arguments.delta":
            yield FunctionFragment(
                name=current_function_name,
                arguments=getattr(event, "delta", "") or "",
            )
            current_function_name = ""  # name only travels with the first fragment

        elif event_type == "response.refusal.done":
            # deliberately not yielded: a refusal must never reach the synthesizer as speech
            logger.warning(
                f"Model refused to respond: {getattr(event, 'refusal', '') or '<empty>'}"
            )

        elif event_type == "response.incomplete":
            # 'max_output_tokens' here means the reasoning pass consumed the budget before
            # producing visible output - the bot goes silent with no other signal
            logger.warning(
                f"Responses stream ended incomplete (reason: {_incomplete_reason(event)})"
            )
            break

        elif event_type == "response.failed":
            logger.error(f"Responses stream failed: {_response_error(event)}")
            break

        elif event_type == "error":
            logger.error(
                "Responses stream emitted an error event: "
                f"code={getattr(event, 'code', None)} message={getattr(event, 'message', None)}"
            )
            break

        elif event_type == "response.completed":
            break
