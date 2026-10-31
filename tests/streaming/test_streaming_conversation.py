import asyncio
import threading
from typing import AsyncGenerator, List, Optional
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pydantic.v1 import BaseModel
from pytest_mock import MockerFixture

from tests.fakedata.conversation import (
    DEFAULT_CHAT_GPT_AGENT_CONFIG,
    DEFAULT_SYNTHESIZER_CONFIG,
    DummyOutputDevice,
    create_fake_agent,
    create_fake_streaming_conversation,
)
from tests.fixtures.synthesizer import TestSynthesizer, TestSynthesizerConfig
from tests.fixtures.transcriber import TestAsyncTranscriber, TestTranscriberConfig
from vocode.streaming.agent.base_agent import AgentResponseMessage, AgentResponseStop
from vocode.streaming.agent.echo_agent import EchoAgent
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.agent import EchoAgentConfig, InterruptSensitivity
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.synthesizer import LiveKitSynthesizerConfig
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.models.transcript import ActionStart, Message, Transcript
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.base_synthesizer import SynthesisResult
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer
from vocode.streaming.synthesizer.livekit_synthesizer import LiveKitSynthesizer
from vocode.streaming.utils.worker import QueueConsumer


class ShouldIgnoreUtteranceTestCase(BaseModel):
    transcript: Transcript
    human_transcription: str
    expected: bool


async def _get_from_consumer_queue_if_exists(queue_consumer: QueueConsumer, timeout: float = 0.1):
    try:
        return await asyncio.wait_for(queue_consumer.input_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def test_interrupt_sensitivity(mocker: MockerFixture):
    from vocode.streaming.streaming_conversation import StreamingConversation

    mock_instance = mocker.MagicMock(spec=StreamingConversation.TranscriptionsWorker)
    mocker.patch.object(
        mock_instance,
        "is_transcription_backchannel",
        wraps=StreamingConversation.TranscriptionsWorker.is_transcription_backchannel,
    )
    mock_instance.conversation = mocker.MagicMock()
    mock_instance.conversation.agent = mocker.MagicMock()

    mock_instance.conversation.agent.get_agent_config = mocker.MagicMock(
        return_value=mocker.MagicMock(interrupt_sensitivity="high")
    )

    test_transcription = Transcription(message="test", confidence=1.0, is_final=True)
    assert mock_instance.is_transcription_backchannel(mock_instance, test_transcription) is False

    mock_instance.conversation.agent.get_agent_config = mocker.MagicMock(
        return_value=mocker.MagicMock(interrupt_sensitivity="low")
    )
    assert mock_instance.is_transcription_backchannel(mock_instance, test_transcription) is True


@pytest.mark.parametrize(
    "test_case",
    [
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[],
            ),
            human_transcription="hi",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                ],
            ),
            human_transcription="hi",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                        is_end_of_turn=False,
                    ),
                ],
            ),
            human_transcription="one two three four",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                        is_end_of_turn=False,
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=True,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                    Message(
                        sender=Sender.HUMAN,
                        text="hi there",
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                    ),
                    ActionStart(
                        action_type="action",
                        action_input=MagicMock(spec=ActionInput),
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=True,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                    ActionStart(
                        action_type="action",
                        action_input=MagicMock(spec=ActionInput),
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=False,
        ),
    ],
)
def test_should_ignore_utterance(
    mocker: MockerFixture,
    test_case: ShouldIgnoreUtteranceTestCase,
):
    from vocode.streaming.streaming_conversation import StreamingConversation

    conversation = mocker.MagicMock()
    transcriptions_worker = StreamingConversation.TranscriptionsWorker(
        conversation=conversation,
        interruptible_event_factory=mocker.MagicMock(),
    )
    conversation = mocker.MagicMock()
    conversation.interrupt_sensitivity = "low"
    transcriptions_worker.has_associated_ignored_utterance = False

    transcriptions_worker.conversation.transcript = test_case.transcript
    assert (
        transcriptions_worker.should_ignore_utterance(
            Transcription(message=test_case.human_transcription, confidence=1.0, is_final=True),
        )
        == test_case.expected
    )


class TranscriptionsWorkerTestCase(BaseModel):
    transcriptions: List[Transcription]
    transcript: Transcript
    should_broadcast_interrupt: bool
    initial_message_ongoing: bool
    should_kick_off_pipeline: bool
    interrupt_sensitivity: InterruptSensitivity


async def _mock_streaming_conversation_constructor(
    mocker: MockerFixture,
    interrupt_sensitivity: InterruptSensitivity = "low",
):

    streaming_conversation = create_fake_streaming_conversation(
        mocker,
        agent=create_fake_agent(
            mocker,
            DEFAULT_CHAT_GPT_AGENT_CONFIG.copy(
                update={"interrupt_sensitivity": interrupt_sensitivity}
            ),
        ),
    )
    streaming_conversation.broadcast_interrupt = mocker.MagicMock(
        wraps=streaming_conversation.broadcast_interrupt
    )
    return streaming_conversation


@pytest.mark.asyncio
async def test_transcriptions_worker_ignores_utterances_before_initial_message(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    transcriptions_worker_consumer = QueueConsumer()
    streaming_conversation.transcriptions_worker.consumer = transcriptions_worker_consumer
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="sup",
            confidence=1.0,
            is_final=True,
        ),
    )
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is",
            confidence=1.0,
            is_final=False,
        ),
    )
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is calling?",
            confidence=1.0,
            is_final=True,
        ),
    )
    streaming_conversation.transcriptions_worker.start()
    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert not streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcript.add_bot_message(
        text="hi there", is_final=True, conversation_id="test"
    )
    streaming_conversation.initial_message_tracker.set()

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is this?",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _get_from_consumer_queue_if_exists(
        transcriptions_worker_consumer
    )
    assert transcription_agent_input.payload.transcription.message == "hi, who is this?"
    assert streaming_conversation.broadcast_interrupt.called

    assert streaming_conversation.transcript.event_logs[0].sender == Sender.BOT

    human_backchannels = streaming_conversation.transcript.event_logs[1:]
    assert all(
        backchannel.sender == Sender.HUMAN and backchannel.is_backchannel
        for backchannel in human_backchannels
    )
    await streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_ignores_associated_ignored_utterance(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.initial_message_tracker.set()
    streaming_conversation.transcript.add_bot_message(
        text="Hi, I was wondering",
        is_final=False,
        conversation_id="test",
    )

    transcriptions_worker_consumer = QueueConsumer()
    streaming_conversation.transcriptions_worker.consumer = transcriptions_worker_consumer
    streaming_conversation.transcriptions_worker.start()

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="i'm listening",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert not streaming_conversation.broadcast_interrupt.called  # ignored for length of response

    streaming_conversation.transcript.event_logs[-1].text = (
        "Hi, I was wondering if you had a chance to look at my email?"
    )
    streaming_conversation.transcript.event_logs[-1].is_final = True
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="I'm listening.",
            confidence=1.0,
            is_final=True,
        ),
    )

    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert not streaming_conversation.broadcast_interrupt.called  # ignored for length of response

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="I have not yet gotten a chance.",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _get_from_consumer_queue_if_exists(
        transcriptions_worker_consumer
    )
    assert (
        transcription_agent_input.payload.transcription.message == "I have not yet gotten a chance."
    )
    assert streaming_conversation.broadcast_interrupt.called
    assert [message.text for message in streaming_conversation.transcript.event_logs] == [
        "Hi, I was wondering if you had a chance to look at my email?",
        "I'm listening.",
    ]
    assert streaming_conversation.transcript.event_logs[-1].is_backchannel
    await streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_interrupts_on_interim_transcripts(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.initial_message_tracker.set()
    streaming_conversation.transcript.add_bot_message(
        text="Hi, I was wondering",
        is_final=False,
        conversation_id="test",
    )

    transcriptions_worker_consumer = QueueConsumer()
    streaming_conversation.transcriptions_worker.consumer = transcriptions_worker_consumer
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, could you stop",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, could you stop talking please?",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _get_from_consumer_queue_if_exists(
        transcriptions_worker_consumer
    )
    assert (
        transcription_agent_input.payload.transcription.message
        == "Sorry, could you stop talking please?"
    )

    assert streaming_conversation.transcript.event_logs[-1].sender == Sender.BOT
    assert streaming_conversation.transcript.event_logs[-1].text == "Hi, I was wondering"
    await streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_interrupts_immediately_before_bot_has_begun_turn(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.initial_message_tracker.set()

    transcriptions_worker_consumer = QueueConsumer()
    streaming_conversation.transcriptions_worker.consumer = transcriptions_worker_consumer
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry,",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, what?",
            confidence=1.0,
            is_final=True,
        ),
    )
    transcription_agent_input = await _get_from_consumer_queue_if_exists(
        transcriptions_worker_consumer
    )
    assert transcription_agent_input.payload.transcription.message == "Sorry, what?"
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcript.add_bot_message(
        text="", is_final=False, conversation_id="test"
    )

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Couldn't",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _get_from_consumer_queue_if_exists(transcriptions_worker_consumer) is None
    assert streaming_conversation.broadcast_interrupt.called

    await streaming_conversation.transcriptions_worker.terminate()


def _create_dummy_synthesis_result(
    message: str = "Hi there",
    num_audio_chunks: int = 3,
    chunk_generator_override: Optional[AsyncGenerator[SynthesisResult.ChunkResult, None]] = None,
):
    async def chunk_generator():
        for i in range(num_audio_chunks):
            yield SynthesisResult.ChunkResult(chunk=b"", is_last_chunk=i == num_audio_chunks - 1)

    def get_message_up_to(seconds: Optional[float]):
        if seconds is None:
            return message
        return message[: len(message) // 2]

    return SynthesisResult(
        chunk_generator=chunk_generator_override or chunk_generator(),
        get_message_up_to=get_message_up_to,
    )


@pytest.mark.asyncio
async def test_send_speech_to_output_uninterrupted(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(mocker)
    synthesis_result = _create_dummy_synthesis_result()
    stop_event = threading.Event()
    transcript_message = Message(
        text="",
        sender=Sender.BOT,
    )

    streaming_conversation.output_device.start()
    message_sent, cut_off = await streaming_conversation.send_speech_to_output(
        message="Hi there",
        synthesis_result=synthesis_result,
        stop_event=stop_event,
        seconds_per_chunk=0.1,
        transcript_message=transcript_message,
    )
    streaming_conversation.output_device.flush()

    assert message_sent == "Hi there"
    assert not cut_off
    assert transcript_message.text == "Hi there"
    assert transcript_message.is_final


@pytest.mark.asyncio
async def test_send_speech_to_output_interrupted_before_all_chunks_sent(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(mocker)
    synthesis_result = _create_dummy_synthesis_result()
    stop_event = threading.Event()
    transcript_message = Message(
        text="",
        sender=Sender.BOT,
    )
    stop_event.set()

    streaming_conversation.output_device.start()
    message_sent, cut_off = await streaming_conversation.send_speech_to_output(
        message="Hi there",
        synthesis_result=synthesis_result,
        stop_event=stop_event,
        seconds_per_chunk=0.1,
        transcript_message=transcript_message,
    )
    streaming_conversation.output_device.flush()

    assert message_sent != "Hi there"
    assert cut_off
    assert transcript_message.text != "Hi there"
    assert not transcript_message.is_final


@pytest.mark.asyncio
async def test_send_speech_to_output_interrupted_during_playback(
    mocker: MockerFixture,
):
    finished_sending_chunks = asyncio.Event()

    async def chunk_generator():
        yield SynthesisResult.ChunkResult(chunk=b"", is_last_chunk=False)
        yield SynthesisResult.ChunkResult(chunk=b"", is_last_chunk=False)
        yield SynthesisResult.ChunkResult(chunk=b"", is_last_chunk=True)
        finished_sending_chunks.set()

    streaming_conversation = await _mock_streaming_conversation_constructor(mocker)
    synthesis_result = _create_dummy_synthesis_result(chunk_generator_override=chunk_generator())
    stop_event = threading.Event()
    transcript_message = Message(
        text="",
        sender=Sender.BOT,
    )

    streaming_conversation.output_device.wait_for_interrupt = True

    streaming_conversation.output_device.start()
    send_speech_to_output_task = asyncio.create_task(
        streaming_conversation.send_speech_to_output(
            message="Hi there",
            synthesis_result=synthesis_result,
            stop_event=stop_event,
            seconds_per_chunk=0.1,
            transcript_message=transcript_message,
        )
    )
    await finished_sending_chunks.wait()
    stop_event.set()
    streaming_conversation.output_device.interrupt_event.set()
    message_sent, cut_off = await send_speech_to_output_task
    await streaming_conversation.output_device.terminate()

    assert message_sent != "Hi there"
    assert cut_off
    assert transcript_message.text != "Hi there"
    assert not transcript_message.is_final


def _livekit_conversation(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    synthesizer = LiveKitSynthesizer(
        LiveKitSynthesizerConfig(
            provider="cartesia",
            streaming=False,
            sampling_rate=8000,
            audio_encoding=AudioEncoding.MULAW,
        )
    )
    streaming_conversation = create_fake_streaming_conversation(mocker, synthesizer=synthesizer)
    streaming_conversation.output_device.start()
    return synthesizer, streaming_conversation


@pytest.mark.asyncio
async def test_livekit_reply_cut_off_before_any_audio_is_recorded_as_unheard(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    synthesizer, streaming_conversation = _livekit_conversation(mocker, monkeypatch)
    turn = synthesizer.current_turn(160)
    turn.add_text("No worries at all, a lot of people are still figuring out what matters most.")
    synthesis_result = synthesizer.synthesis_result(turn)
    transcript_message = Message(text="", sender=Sender.BOT)
    stop_event = threading.Event()
    stop_event.set()
    synthesizer.cancel_turn()

    message_sent, cut_off = await streaming_conversation.send_speech_to_output(
        message=turn.text,
        synthesis_result=synthesis_result,
        stop_event=stop_event,
        seconds_per_chunk=0.1,
        transcript_message=transcript_message,
    )
    streaming_conversation.output_device.flush()

    assert cut_off
    assert message_sent == ""
    assert transcript_message.text == ""
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_livekit_reply_cut_off_while_its_next_sentence_generates_keeps_only_what_played(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    synthesizer, streaming_conversation = _livekit_conversation(mocker, monkeypatch)
    turn = synthesizer.current_turn(160)
    synthesizer.push_cached(turn, "Hi.", b"\xff" * 320)
    # the next sentence was sent to the provider, but none of its audio has arrived yet
    turn.add_text("This part was never spoken.")
    synthesis_result = synthesizer.synthesis_result(turn)
    transcript_message = Message(text="", sender=Sender.BOT)
    stop_event = threading.Event()

    playback = asyncio.create_task(
        streaming_conversation.send_speech_to_output(
            message=turn.text,
            synthesis_result=synthesis_result,
            stop_event=stop_event,
            seconds_per_chunk=0.1,
            transcript_message=transcript_message,
        )
    )
    for _ in range(2):
        await asyncio.wait_for(streaming_conversation.output_device.dummy_playback_queue.get(), 1)
    stop_event.set()
    synthesizer.cancel_turn()
    message_sent, cut_off = await asyncio.wait_for(playback, 1)

    assert cut_off
    assert "never spoken" not in transcript_message.text
    assert not transcript_message.is_final
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_livekit_reply_whose_stream_failed_before_any_audio_is_recorded_as_unheard(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    synthesizer, streaming_conversation = _livekit_conversation(mocker, monkeypatch)
    turn = synthesizer.current_turn(160)
    turn.add_text("This reply never reached the caller.")
    synthesis_result = synthesizer.synthesis_result(turn)
    # what read_stream does when the provider errors: end the audio without completing it
    turn.audio.put_nowait(None)
    transcript_message = Message(text="", sender=Sender.BOT)

    await streaming_conversation.send_speech_to_output(
        message=turn.text,
        synthesis_result=synthesis_result,
        stop_event=threading.Event(),
        seconds_per_chunk=0.1,
        transcript_message=transcript_message,
    )
    streaming_conversation.output_device.flush()

    assert transcript_message.text == ""
    await synthesizer.tear_down()


@pytest_asyncio.fixture
async def input_streaming_conversation(mocker: MockerFixture):
    conversations: List[StreamingConversation] = []

    def create(spec=LiveKitSynthesizer):
        synthesizer = mocker.MagicMock(spec=spec)
        synthesizer.get_synthesizer_config = mocker.MagicMock(
            return_value=DEFAULT_SYNTHESIZER_CONFIG
        )
        streaming_conversation = create_fake_streaming_conversation(mocker, synthesizer=synthesizer)
        streaming_conversation.agent_responses_worker.consumer = QueueConsumer()
        streaming_conversation.agent_responses_worker.start()
        conversations.append(streaming_conversation)
        return synthesizer, streaming_conversation

    yield create
    for streaming_conversation in conversations:
        await streaming_conversation.agent_responses_worker.terminate()


async def _send_agent_message(
    streaming_conversation: StreamingConversation,
    message: BaseMessage,
    is_first: bool = True,
    is_interruptible: bool = True,
):
    worker = streaming_conversation.agent_responses_worker
    worker.consume_nonblocking(
        worker.interruptible_event_factory.create_interruptible_agent_response_event(
            AgentResponseMessage(message=message, is_first=is_first),
            is_interruptible=is_interruptible,
        )
    )
    while not worker._input_queue.empty() or worker.current_task is not None:
        await asyncio.sleep(0)


def _queued_synthesis_results(streaming_conversation: StreamingConversation):
    queue = streaming_conversation.agent_responses_worker.consumer.input_queue
    results = []
    while not queue.empty():
        results.append(queue.get_nowait().payload[1])
    return results


@pytest.mark.asyncio
async def test_input_streaming_reply_is_played_after_previous_reply_interrupted_before_its_end_of_turn(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation()

    interrupted_reply = mocker.MagicMock()
    synthesizer.get_current_utterance_synthesis_result.return_value = interrupted_reply
    await _send_agent_message(
        streaming_conversation, LLMToken(text="I'm really sorry to hear that.")
    )
    await streaming_conversation.broadcast_interrupt()

    next_reply = mocker.MagicMock()
    synthesizer.get_current_utterance_synthesis_result.return_value = next_reply
    await _send_agent_message(
        streaming_conversation, LLMToken(text="Is there an area you're hoping to move to?")
    )

    assert _queued_synthesis_results(streaming_conversation) == [interrupted_reply, next_reply]
    synthesizer.cancel_turn.assert_called_once()


@pytest.mark.asyncio
async def test_caller_speaking_over_an_uninterruptible_livekit_turn_lets_all_of_it_play(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation()
    turn = mocker.MagicMock()
    synthesizer.get_current_utterance_synthesis_result.return_value = turn
    await _send_agent_message(
        streaming_conversation,
        LLMToken(text="Great, let me connect you with Sarah."),
        is_interruptible=False,
    )

    await streaming_conversation.broadcast_interrupt()
    # the rest of the same reply joins its turn
    await _send_agent_message(
        streaming_conversation,
        LLMToken(text="She'll be right with you."),
        is_first=False,
        is_interruptible=False,
    )
    # the agent's next reply opens a new turn, although the last one never got its EndOfTurn
    next_reply = mocker.MagicMock()
    synthesizer.get_current_utterance_synthesis_result.return_value = next_reply
    await _send_agent_message(streaming_conversation, LLMToken(text="Anything else?"))

    synthesizer.cancel_turn.assert_not_called()
    synthesizer.handle_end_of_turn.assert_awaited()
    assert _queued_synthesis_results(streaming_conversation) == [turn, next_reply]


@pytest.mark.asyncio
async def test_caller_speaking_while_the_next_reply_loads_does_not_cancel_that_reply(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation()
    synthesizer.get_current_utterance_synthesis_result.return_value = mocker.MagicMock()
    await _send_agent_message(streaming_conversation, LLMToken(text="One."))
    await streaming_conversation.broadcast_interrupt()

    cache_lookup_done = asyncio.Event()

    async def look_up_cache(*args, **kwargs):
        await cache_lookup_done.wait()
        return mocker.MagicMock()

    synthesizer.create_speech = mocker.AsyncMock(side_effect=look_up_cache)
    worker = streaming_conversation.agent_responses_worker
    worker.consume_nonblocking(
        worker.interruptible_event_factory.create_interruptible_agent_response_event(
            AgentResponseMessage(message=BaseMessage(text="Sorry, go ahead."), is_first=True),
            is_interruptible=False,
        )
    )
    while not synthesizer.create_speech.called:
        await asyncio.sleep(0)
    await streaming_conversation.broadcast_interrupt()
    cache_lookup_done.set()

    # only the first, interrupted turn was cancelled
    synthesizer.cancel_turn.assert_called_once()


@pytest.mark.asyncio
async def test_input_streaming_reply_after_a_filler_is_played_as_its_own_message(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation()
    filler = mocker.MagicMock()
    reply = mocker.MagicMock()
    synthesizer.create_speech = mocker.AsyncMock(return_value=filler)
    synthesizer.get_current_utterance_synthesis_result.return_value = reply

    await _send_agent_message(streaming_conversation, BotBackchannel(text="Got it, one moment."))
    await _send_agent_message(
        streaming_conversation, LLMToken(text="We have three homes there."), is_first=False
    )

    assert _queued_synthesis_results(streaming_conversation) == [filler, reply]


@pytest.mark.asyncio
async def test_other_input_streaming_voices_keep_vocodes_filler_flow(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation(
        spec=InputStreamingSynthesizer
    )
    filler = mocker.MagicMock()
    synthesizer.create_speech = mocker.AsyncMock(return_value=filler)

    await _send_agent_message(streaming_conversation, BotBackchannel(text="Got it, one moment."))
    await _send_agent_message(
        streaming_conversation, LLMToken(text="We have three homes there."), is_first=False
    )

    assert _queued_synthesis_results(streaming_conversation) == [filler]


@pytest.mark.asyncio
async def test_goodbye_hangs_up_once_the_whole_livekit_turn_has_played(
    mocker: MockerFixture, input_streaming_conversation
):
    synthesizer, streaming_conversation = input_streaming_conversation()
    synthesizer.handle_end_of_turn = mocker.AsyncMock()
    mark_terminated = mocker.patch.object(streaming_conversation, "mark_terminated")
    worker = streaming_conversation.agent_responses_worker

    await _send_agent_message(streaming_conversation, LLMToken(text="Thanks for your time."))
    await _send_agent_message(
        streaming_conversation, LLMToken(text="Have a great day, goodbye."), is_first=False
    )
    turn_played = worker.consumer.input_queue.get_nowait().agent_response_tracker
    synthesizer.handle_end_of_turn.reset_mock()
    stop = asyncio.create_task(
        worker.process(
            worker.interruptible_event_factory.create_interruptible_agent_response_event(
                AgentResponseStop()
            )
        )
    )
    await asyncio.sleep(0)

    synthesizer.handle_end_of_turn.assert_awaited_once()
    mark_terminated.assert_not_called()
    turn_played.set()
    await asyncio.wait_for(stop, timeout=1)
    mark_terminated.assert_called_once_with(bot_disconnect=True)


@pytest.mark.asyncio
async def test_streaming_conversation_pipeline(
    mocker: MockerFixture,
):
    output_device = DummyOutputDevice(sampling_rate=48000, audio_encoding=AudioEncoding.LINEAR16)
    streaming_conversation = StreamingConversation(
        output_device=output_device,
        transcriber=TestAsyncTranscriber(
            TestTranscriberConfig(
                sampling_rate=48000,
                audio_encoding=AudioEncoding.LINEAR16,
                chunk_size=480,
            )
        ),
        agent=EchoAgent(
            EchoAgentConfig(initial_message=BaseMessage(text="Hi there")),
        ),
        synthesizer=TestSynthesizer(TestSynthesizerConfig.from_output_device(output_device)),
    )
    await streaming_conversation.start()
    await streaming_conversation.initial_message_tracker.wait()
    streaming_conversation.receive_audio(b"test")
    initial_message_audio_chunk = await output_device.dummy_playback_queue.get()
    assert initial_message_audio_chunk.data == b"Hi there"
    first_response_audio_chunk = await output_device.dummy_playback_queue.get()
    assert first_response_audio_chunk.data == b"test"
