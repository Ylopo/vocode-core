import asyncio
import base64
import json
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CartesiaSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.telephony.constants import MULAW_SILENCE_BYTE, PCM_SILENCE_BYTE
from vocode.streaming.utils import get_chunk_size_per_second
from vocode.streaming.utils.create_task import asyncio_create_task

CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_WEBSOCKET_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2025-04-16"
MAX_WEBSOCKET_AGE_SECONDS = 300
AUDIO_TIMEOUT_SECONDS = 10
WORDS_PER_MINUTE = 150


class _Sentence:
    def __init__(self, start_seconds: float):
        self.audio: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        # Cartesia's timestamps run across the whole context, so they are stored as sent
        # and made relative to the sentence's own audio when read
        self.start_seconds = start_seconds
        self.word_start_times: List[Tuple[str, float]] = []

    def finish(self):
        self.audio.put_nowait(None)


class _Context:
    """A Cartesia context carries one bot turn, so prosody continues across its sentences."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.pending_sentences: Deque[_Sentence] = deque()
        self.audio_seconds = 0.0
        self.closed = False

    def add_sentence(self) -> _Sentence:
        sentence = _Sentence(start_seconds=self.audio_seconds)
        self.pending_sentences.append(sentence)
        return sentence

    def finish_sentence(self):
        if self.pending_sentences:
            self.pending_sentences.popleft().finish()
        if self.pending_sentences:
            self.pending_sentences[0].start_seconds = self.audio_seconds

    def close(self):
        self.closed = True
        while self.pending_sentences:
            self.pending_sentences.popleft().finish()


class CartesiaSynthesizer(BaseSynthesizer[CartesiaSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: CartesiaSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.api_key = getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Cartesia API key")

        if synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            encoding = "pcm_mulaw"
            self.silence_byte = MULAW_SILENCE_BYTE
        elif synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            encoding = "pcm_s16le"
            self.silence_byte = PCM_SILENCE_BYTE
        else:
            raise ValueError(f"Unsupported audio encoding {synthesizer_config.audio_encoding}")
        self.output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": synthesizer_config.sampling_rate,
        }
        self.bytes_per_second = get_chunk_size_per_second(
            synthesizer_config.audio_encoding, synthesizer_config.sampling_rate
        )

        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.websocket_opened_at = 0.0
        self.websocket_lock = asyncio.Lock()
        self.receive_task: Optional[asyncio.Task] = None
        self.contexts: Dict[str, _Context] = {}
        self.current_context: Optional[_Context] = None

    def get_request_body(self, transcript: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model_id": self.synthesizer_config.model_id,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self.synthesizer_config.voice_id},
            "output_format": self.output_format,
            "language": self.synthesizer_config.language,
        }
        generation_config = {
            key: value
            for key, value in (
                ("speed", self.synthesizer_config.speed),
                ("volume", self.synthesizer_config.volume),
                ("emotion", self.synthesizer_config.emotion),
            )
            if value is not None
        }
        if generation_config:
            body["generation_config"] = generation_config
        return body

    def get_websocket_request(
        self, transcript: str, context: _Context, continue_: bool
    ) -> Dict[str, Any]:
        return {
            **self.get_request_body(transcript),
            "context_id": context.id,
            "continue": continue_,
            # each sentence is flushed so its audio ends with a flush_done marker, which is
            # how audio is split back into one SynthesisResult per sentence
            "flush": continue_,
            "add_timestamps": True,
            # sentences arrive complete, so there is nothing to gain from Cartesia buffering text
            "max_buffer_delay_ms": 0,
        }

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        if is_sole_text_chunk:
            return self.create_speech_over_http(message, chunk_size)
        return await self.create_speech_over_websocket(message, chunk_size, is_first_text_chunk)

    def create_speech_over_http(self, message: BaseMessage, chunk_size: int) -> SynthesisResult:
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task(self.stream_bytes(message.text, audio_queue))
        return SynthesisResult(
            self.chunk_generator(audio_queue, chunk_size),
            lambda seconds: self.get_message_cutoff_from_voice_speed(
                message, seconds, WORDS_PER_MINUTE
            ),
        )

    async def stream_bytes(self, text: str, audio_queue: asyncio.Queue[Optional[bytes]]):
        try:
            async with self.async_requestor.get_session().post(
                f"{CARTESIA_BASE_URL}/tts/bytes",
                headers={"X-API-Key": self.api_key, "Cartesia-Version": CARTESIA_API_VERSION},
                json=self.get_request_body(text),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if not response.ok:
                    logger.error(
                        f"Cartesia /tts/bytes returned {response.status}: {await response.text()}"
                    )
                    return
                async for audio in response.content.iter_any():
                    audio_queue.put_nowait(audio)
        except Exception:
            logger.exception("Cartesia /tts/bytes request failed")
        finally:
            audio_queue.put_nowait(None)

    async def create_speech_over_websocket(
        self, message: BaseMessage, chunk_size: int, is_first_text_chunk: bool
    ) -> SynthesisResult:
        websocket = await self.get_websocket(allow_refresh=is_first_text_chunk)
        context = self.current_context
        # Cartesia closes a context on its own after a few idle seconds, so a turn may span contexts
        if is_first_text_chunk or context is None or context.closed:
            if context is not None and not context.closed:
                await self.close_context(context)
            context = _Context()
            self.contexts[context.id] = context
            self.current_context = context

        sentence = context.add_sentence()
        transcript = message.text if message.text.endswith(" ") else message.text + " "
        try:
            await websocket.send_str(
                json.dumps(self.get_websocket_request(transcript, context, continue_=True))
            )
        except Exception:
            logger.exception("Failed to send text to Cartesia")
            context.close()

        return SynthesisResult(
            self.chunk_generator(sentence.audio, chunk_size),
            lambda seconds: self.get_message_up_to(message, sentence, seconds),
        )

    async def get_websocket(self, allow_refresh: bool) -> aiohttp.ClientWebSocketResponse:
        async with self.websocket_lock:
            is_stale = time.monotonic() - self.websocket_opened_at > MAX_WEBSOCKET_AGE_SECONDS
            has_pending_audio = any(context.pending_sentences for context in self.contexts.values())
            if self.websocket is not None and not self.websocket.closed:
                if not (allow_refresh and is_stale and not has_pending_audio):
                    return self.websocket
                await self.websocket.close()

            websocket = await self.async_requestor.get_session().ws_connect(
                f"{CARTESIA_WEBSOCKET_URL}?cartesia_version={CARTESIA_API_VERSION}",
                headers={"X-API-Key": self.api_key},
            )
            self.websocket = websocket
            self.websocket_opened_at = time.monotonic()
            self.receive_task = asyncio_create_task(self.receive_messages(websocket))
            return websocket

    async def receive_messages(self, websocket: aiohttp.ClientWebSocketResponse):
        try:
            async for message in websocket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(message.data)
                # cancelled contexts keep sending until Cartesia stops them, so unknown ids are dropped
                context = self.contexts.get(data.get("context_id"))
                if context is not None:
                    self.handle_message(context, data)
        except Exception:
            logger.exception("Error receiving from Cartesia")
        finally:
            if self.websocket is websocket:
                for context in self.contexts.values():
                    context.close()
                self.contexts.clear()
                self.current_context = None

    def handle_message(self, context: _Context, data: Dict[str, Any]):
        message_type = data.get("type")
        if message_type == "chunk":
            audio = base64.b64decode(data["data"])
            if context.pending_sentences:
                context.pending_sentences[0].audio.put_nowait(audio)
            context.audio_seconds += len(audio) / self.bytes_per_second
        elif message_type == "timestamps":
            if context.pending_sentences:
                word_timestamps = data["word_timestamps"]
                context.pending_sentences[0].word_start_times.extend(
                    zip(word_timestamps["words"], word_timestamps["start"])
                )
        elif message_type == "flush_done":
            context.finish_sentence()
        elif message_type in ("done", "error"):
            if message_type == "error":
                logger.error(f"Cartesia error for context {context.id}: {data}")
            context.close()
            self.contexts.pop(context.id, None)

    async def close_context(self, context: _Context):
        if self.websocket is None or self.websocket.closed:
            return
        try:
            await self.websocket.send_str(
                json.dumps(self.get_websocket_request(" ", context, continue_=False))
            )
        except Exception:
            logger.exception("Failed to close Cartesia context")

    async def cancel_context(self, context: _Context):
        self.contexts.pop(context.id, None)
        context.close()
        if self.websocket is None or self.websocket.closed:
            return
        try:
            await self.websocket.send_str(json.dumps({"context_id": context.id, "cancel": True}))
        except Exception:
            logger.exception("Failed to cancel Cartesia context")

    async def chunk_generator(
        self, audio_queue: asyncio.Queue[Optional[bytes]], chunk_size: int
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        buffer = bytearray()
        while True:
            try:
                audio = await asyncio.wait_for(audio_queue.get(), timeout=AUDIO_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for audio from Cartesia")
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

    def get_message_up_to(
        self, message: BaseMessage, sentence: _Sentence, seconds: Optional[float]
    ) -> str:
        if seconds is None:
            return message.text
        if not sentence.word_start_times:
            return self.get_message_cutoff_from_voice_speed(message, seconds, WORDS_PER_MINUTE)
        return " ".join(
            word
            for word, start in sentence.word_start_times
            if start - sentence.start_seconds < seconds
        )

    def ready_synthesizer(self, chunk_size: int):
        # Called once the human has finished speaking: any turn still generating was talked over
        for context in list(self.contexts.values()):
            if context.pending_sentences:
                asyncio_create_task(self.cancel_context(context))
        asyncio_create_task(self.get_websocket(allow_refresh=True))

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: CartesiaSynthesizerConfig):
        return ":".join(
            (
                "cartesia",
                synthesizer_config.voice_id,
                synthesizer_config.model_id,
                synthesizer_config.language,
                str(synthesizer_config.speed),
                str(synthesizer_config.volume),
                str(synthesizer_config.emotion),
                synthesizer_config.audio_encoding,
            )
        )

    async def tear_down(self):
        await super().tear_down()
        if self.receive_task is not None:
            self.receive_task.cancel()
        if self.websocket is not None and not self.websocket.closed:
            await self.websocket.close()
