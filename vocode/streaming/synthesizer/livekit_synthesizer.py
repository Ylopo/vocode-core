import asyncio
import audioop
import importlib
import inspect
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from livekit.agents import tts
from livekit.agents.tokenize import SentenceStream, SentenceTokenizer, TokenData
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.agents.utils import is_given
from loguru import logger

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.synthesizer import LiveKitSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer
from vocode.streaming.telephony.constants import MULAW_SILENCE_BYTE, PCM_SILENCE_BYTE
from vocode.streaming.utils import get_chunk_size_per_second
from vocode.streaming.utils.create_task import asyncio_create_task

AUDIO_TIMEOUT_SECONDS = 10
WORDS_PER_MINUTE = 150
# Set by vocode for every plugin that accepts them, so they cannot be given in options
MANAGED_OPTIONS = {"sample_rate", "use_websocket", "tokenizer", "http_session", "api_key"}
# Plugins send our API key to the address in their *_url options, so those may only point at the
# provider itself. A provider missing here cannot have its address changed at all.
PROVIDER_DOMAINS = {"cartesia": "cartesia.ai", "elevenlabs": "elevenlabs.io", "rime": "rime.ai"}
# Rime streams its current models over WebSocket v1; its legacy ones (mist, mistv2) only over /ws3
RIME_WEBSOCKET_V1_URLS = {
    "coda": "wss://coda.api.rime.ai/ws",
    "mistv3": "wss://mist.api.rime.ai/ws",
}


def _ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _check_endpoint_options(provider: str, options: Dict[str, Any]):
    if "allow_custom_endpoint" in options:
        raise ValueError("Remove allow_custom_endpoint from options")
    domain = PROVIDER_DOMAINS.get(provider)
    for name, value in options.items():
        if not name.endswith("_url"):
            continue
        url = urlparse(value) if isinstance(value, str) else None
        host = (url.hostname or "") if url else ""
        on_provider = domain is not None and (host == domain or host.endswith(f".{domain}"))
        if not (url and url.scheme in ("https", "wss") and on_provider):
            raise ValueError(
                f"{name} must be an https or wss address on {domain}"
                if domain
                else f"{name} cannot be set for {provider}"
            )


def _with_rime_websocket_v1(options: Dict[str, Any]) -> Dict[str, Any]:
    if options.keys() & {"base_url", "websocket_url"}:
        return options
    model = options.get("model", "coda")
    url = RIME_WEBSOCKET_V1_URLS.get(model)
    return {**options, "model": model, "websocket_url": url} if url else options


class _WholeSentenceStream(SentenceStream):
    def __init__(self):
        super().__init__()
        self.segment_id = uuid.uuid4().hex

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if text.strip():
            self._event_ch.send_nowait(TokenData(segment_id=self.segment_id, token=text.strip()))

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        self._do_close()

    async def aclose(self) -> None:
        self._do_close()


class _WholeSentenceTokenizer(SentenceTokenizer):
    """Treats every pushed text as one finished sentence. The agent has already split the LLM
    output into sentences, and a second tokenizer would hold each one back until the next began."""

    def tokenize(self, text: str, *, language: Optional[str] = None) -> List[str]:
        return [text]

    def stream(self, *, language: Optional[str] = None) -> SentenceStream:
        return _WholeSentenceStream()


def build_plugin_tts(
    provider: str,
    options: Dict[str, Any],
    streaming: bool,
    sampling_rate: int,
    http_session: Optional[aiohttp.ClientSession] = None,
) -> tts.TTS:
    """Creates the provider's LiveKit TTS. Raises ValueError for anything the plugin would reject."""
    managed = sorted(MANAGED_OPTIONS & options.keys())
    if managed:
        raise ValueError(f"Remove {', '.join(managed)} from options; vocode sets these itself")
    _check_endpoint_options(provider, options)
    if provider == "rime" and streaming:
        options = _with_rime_websocket_v1(options)
    try:
        tts_class = importlib.import_module(f"livekit.plugins.{provider}").TTS
    except (ImportError, AttributeError):
        raise ValueError(f"No LiveKit text-to-speech plugin is installed for {provider!r}")

    parameters = inspect.signature(tts_class).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        unknown = sorted(options.keys() - parameters.keys())
        if unknown:
            allowed = sorted(parameters.keys() - MANAGED_OPTIONS)
            raise ValueError(
                f"Unknown {provider} options: {', '.join(unknown)}. Allowed: {', '.join(allowed)}"
            )

    generic = {
        "sample_rate": sampling_rate,
        # a websocket_url turns streaming on by itself, and plugins reject both together
        "use_websocket": None if "websocket_url" in options else streaming,
        "tokenizer": _WholeSentenceTokenizer(),
        "http_session": http_session,
    }
    kwargs = {**options, **{k: v for k, v in generic.items() if k in parameters and v is not None}}
    try:
        plugin_tts = tts_class(**kwargs)
    except Exception as e:
        raise ValueError(f"Invalid {provider} options: {e}") from e
    if streaming and not plugin_tts.capabilities.streaming:
        raise ValueError(
            f"The {provider} plugin does not support streaming; set streaming to false"
        )
    return plugin_tts


class _Stream:
    """The audio of a whole bot turn, or of a single one-off message. Leading messages that are
    in the audio cache play from it; everything after them goes through one plugin stream."""

    def __init__(self, label: str, chunk_size: int):
        self.label = label
        self.chunk_size = chunk_size
        self.audio: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.text = ""
        self.word_start_times: List[Tuple[str, float]] = []
        self.input_ended = False
        self.audio_complete = False
        self.started_at = time.monotonic()
        self.first_audio_ms: Optional[int] = None
        self.audio_bytes = 0
        self.plugin_stream: Optional[tts.SynthesizeStream] = None
        self.reader: Optional[asyncio.Task] = None
        # added to the plugin's word timestamps so they count from the start of this stream's audio
        self.time_offset = 0.0

    def add_text(self, text: str):
        self.text = f"{self.text} {text}".strip()


class LiveKitSynthesizer(BaseSynthesizer[LiveKitSynthesizerConfig], InputStreamingSynthesizer):
    """Runs any LiveKit TTS plugin the way LiveKit's agents do: one stream per bot turn,
    with the turn's sentences pushed into it as they are generated."""

    def __init__(self, synthesizer_config: LiveKitSynthesizerConfig):
        super().__init__(synthesizer_config)
        self.label = f"LiveKit [{synthesizer_config.provider}]"
        self.silence_byte = (
            MULAW_SILENCE_BYTE
            if synthesizer_config.audio_encoding == AudioEncoding.MULAW
            else PCM_SILENCE_BYTE
        )
        self.bytes_per_second = get_chunk_size_per_second(
            synthesizer_config.audio_encoding, synthesizer_config.sampling_rate
        )
        self.tts: Optional[tts.TTS] = None
        self.streaming_tts: Optional[tts.TTS] = None
        # the plugin keeps its session for the whole call, and the shared AsyncRequestor session
        # can be closed by other code (e.g. the transfer action) while it is still in use
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.turn: Optional[_Stream] = None
        self.stream_count = 0
        self.word_timestamps = False
        self.torn_down = False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        # built during call setup, so the caller's first reply does not wait for it
        self.get_streaming_tts()

    def get_streaming_tts(self) -> tts.TTS:
        # needs a running event loop, since plugins take an aiohttp session
        if self.streaming_tts is None:
            c = self.synthesizer_config
            self.http_session = aiohttp.ClientSession()
            self.tts = build_plugin_tts(
                c.provider,
                c.options,
                c.streaming,
                c.sampling_rate,
                self.http_session,
            )
            # without streaming, LiveKit's adapter sends each sentence as its own (HTTP) request
            self.streaming_tts = (
                self.tts
                if c.streaming
                else tts.StreamAdapter(tts=self.tts, sentence_tokenizer=_WholeSentenceTokenizer())
            )
            # the HTTP adapter only reports sentence-level timings, which are not usable for cut-offs
            self.word_timestamps = c.streaming and self.tts.capabilities.aligned_transcript
            logger.info(
                f"{self.label}: plugin ready (model {self.tts.model}, "
                f"{'streaming' if c.streaming else 'one HTTP request per sentence'})"
            )
        return self.streaming_tts

    def open_stream(self, kind: str, chunk_size: int) -> _Stream:
        self.stream_count += 1
        stream = _Stream(f"{self.label} {kind} {self.stream_count}", chunk_size)
        logger.info(f"{stream.label}: started")
        return stream

    def push_live(self, stream: _Stream, text: str):
        if stream.plugin_stream is None:
            stream.plugin_stream = self.get_streaming_tts().stream()
            stream.time_offset = stream.audio_bytes / self.bytes_per_second
            stream.reader = asyncio_create_task(self.read_stream(stream))
        self.total_chars += len(text)
        stream.add_text(text)
        stream.plugin_stream.push_text(text.rstrip() + " ")
        logger.info(f"{stream.label}: pushing {text!r}")

    def push_cached(self, stream: _Stream, text: str, audio: bytes):
        stream.add_text(text)
        stream.first_audio_ms = stream.first_audio_ms or _ms_since(stream.started_at)
        stream.audio_bytes += len(audio)
        stream.audio.put_nowait(audio)
        logger.info(f"{stream.label}: playing cached audio for {text!r}")

    def end_input(self, stream: _Stream):
        if stream.input_ended:
            return
        stream.input_ended = True
        if stream.plugin_stream is not None:
            stream.plugin_stream.end_input()
        else:
            stream.audio_complete = True
            stream.audio.put_nowait(None)

    async def read_stream(self, stream: _Stream):
        assert stream.plugin_stream is not None
        c = self.synthesizer_config
        resample_state: Any = None
        try:
            async for event in stream.plugin_stream:
                # plugins output mono 16-bit PCM, at the requested rate when they support one
                audio = event.frame.data.tobytes()
                if event.frame.sample_rate != c.sampling_rate:
                    audio, resample_state = audioop.ratecv(
                        audio, 2, 1, event.frame.sample_rate, c.sampling_rate, resample_state
                    )
                if c.audio_encoding == AudioEncoding.MULAW:
                    audio = audioop.lin2ulaw(audio, 2)
                frame_start = stream.audio_bytes / self.bytes_per_second
                stream.first_audio_ms = stream.first_audio_ms or _ms_since(stream.started_at)
                stream.audio_bytes += len(audio)
                stream.audio.put_nowait(audio)
                timed_words = event.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT) or []
                for word in timed_words if self.word_timestamps else []:
                    if not is_given(word.start_time):
                        continue
                    start = word.start_time + stream.time_offset
                    if stream.word_start_times and start < stream.word_start_times[-1][1]:
                        # some providers (e.g. Rime) restart their clock at every sentence
                        stream.time_offset = frame_start - word.start_time
                        start = frame_start
                    stream.word_start_times.append((str(word).strip(), start))
            stream.audio_complete = True
            logger.info(
                f"{stream.label}: done, first audio after {stream.first_audio_ms} ms, "
                f"{stream.audio_bytes / self.bytes_per_second:.2f}s of audio in "
                f"{_ms_since(stream.started_at)} ms"
            )
        except Exception:
            logger.exception(f"{stream.label}: plugin stream failed")
        finally:
            stream.audio.put_nowait(None)
        await stream.plugin_stream.aclose()

    def synthesis_result(self, stream: _Stream) -> SynthesisResult:
        return SynthesisResult(
            self.chunk_generator(stream.audio, stream.chunk_size, stream.label),
            lambda seconds: self.get_message_up_to(stream, seconds),
        )

    def get_message_up_to(self, stream: _Stream, seconds: Optional[float]) -> str:
        if seconds is None:
            return stream.text
        if stream.word_start_times:
            return " ".join(word for word, start in stream.word_start_times if start < seconds)
        if stream.audio_complete and stream.audio_bytes:
            # without word timings, assume the text is spoken at an even pace over the real audio
            heard = len(stream.text) * seconds * self.bytes_per_second / stream.audio_bytes
            words: List[str] = []
            for word in stream.text.split():
                if len(" ".join(words)) >= heard:
                    break
                words.append(word)
            return " ".join(words)
        return self.get_message_cutoff_from_voice_speed(
            BaseMessage(text=stream.text), seconds, WORDS_PER_MINUTE
        )

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        if is_sole_text_chunk or isinstance(message, BotBackchannel):
            return await super().create_speech(
                message, chunk_size, is_first_text_chunk, is_sole_text_chunk
            )
        turn = self.current_turn(chunk_size, new_turn=is_first_text_chunk)
        # cached audio can only lead the turn: once live audio is streaming, the rest stays live
        cached = await self.get_cached_audio(message) if turn.plugin_stream is None else None
        if cached is not None:
            self.push_cached(turn, message.text, cached.audio_data)
        else:
            self.push_live(turn, message.text)
        return self.synthesis_result(turn)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        stream = self.open_stream("one-off", chunk_size)
        self.push_live(stream, message.text)
        self.end_input(stream)
        return self.synthesis_result(stream)

    async def send_token_to_synthesizer(self, message: LLMToken, chunk_size: int):
        self.push_live(self.current_turn(chunk_size), message.text)

    def current_turn(self, chunk_size: int, new_turn: bool = False) -> _Stream:
        if new_turn or self.turn is None or self.turn.input_ended:
            if self.turn is not None:
                self.end_input(self.turn)
            self.turn = self.open_stream("turn", chunk_size)
        return self.turn

    def get_current_utterance_synthesis_result(self) -> SynthesisResult:
        assert self.turn is not None
        return self.synthesis_result(self.turn)

    async def handle_end_of_turn(self):
        if self.turn is not None and not self.turn.input_ended:
            self.end_input(self.turn)
            logger.info(f"{self.turn.label}: all text sent")

    def ready_synthesizer(self, chunk_size: int):
        # Called once the human has finished speaking: a turn still generating was talked over,
        # unless it could not be interrupted
        turn = self.turn
        interrupted = self.streaming_conversation.current_transcription_is_interrupt
        if interrupted and turn is not None and turn.reader is not None and not turn.reader.done():
            asyncio_create_task(self.cancel_stream(turn))
        # only the streaming connection can be opened ahead; the HTTP mode never uses it
        if self.synthesizer_config.streaming and not self.torn_down:
            self.get_streaming_tts().prewarm()

    async def cancel_stream(self, stream: _Stream):
        logger.info(f"{stream.label}: caller spoke over it, cancelling")
        stream.input_ended = True
        if stream.reader is not None:
            stream.reader.cancel()
        if stream.plugin_stream is not None:
            await stream.plugin_stream.aclose()
        stream.audio.put_nowait(None)

    async def chunk_generator(
        self, audio_queue: asyncio.Queue[Optional[bytes]], chunk_size: int, label: str
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        buffer = bytearray()
        while True:
            try:
                audio = await asyncio.wait_for(audio_queue.get(), timeout=AUDIO_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error(f"{label}: no audio received for {AUDIO_TIMEOUT_SECONDS}s, giving up")
                break
            if audio is None:
                break
            buffer.extend(audio)
            while len(buffer) >= chunk_size:
                yield SynthesisResult.ChunkResult(
                    chunk=bytes(buffer[:chunk_size]), is_last_chunk=False
                )
                del buffer[:chunk_size]
        if buffer:
            # the conversation counts every chunk as exactly chunk_size of playback, so pad the last one
            buffer.extend(self.silence_byte * (chunk_size - len(buffer)))
            yield SynthesisResult.ChunkResult(chunk=bytes(buffer), is_last_chunk=True)

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: LiveKitSynthesizerConfig):
        c = synthesizer_config
        options = json.dumps(c.options, sort_keys=True)
        return f"livekit:{c.provider}:{c.streaming}:{options}:{c.audio_encoding.value}"

    async def tear_down(self):
        self.torn_down = True
        await super().tear_down()
        if self.turn is not None and self.turn.reader is not None and not self.turn.reader.done():
            await self.cancel_stream(self.turn)
        if self.streaming_tts is not None and self.streaming_tts is not self.tts:
            await self.streaming_tts.aclose()
        if self.tts is not None:
            await self.tts.aclose()
        if self.http_session is not None:
            await self.http_session.close()
        logger.info(f"{self.label}: torn down after synthesizing {self.total_chars} characters")
