import asyncio
import base64
import json
import uuid
from typing import Optional

import websockets
from loguru import logger

from vocode import getenv
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.models.synthesizer import RIME_DEFAULT_SPEED_ALPHA, RimeSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer
from vocode.streaming.synthesizer.rime_synthesizer import RimeSynthesizer
from vocode.streaming.utils.create_task import asyncio_create_task

RIME_CODA_WS_URL = "wss://api.rime.ai/coda/ws"
RIME_CODA_SUBPROTOCOL = "rime.v1.json"

SENTENCE_ENDINGS = (".", "!", "?")


class RimeWebsocketError(Exception):
    pass


def snap_to_word_boundary(text: str, index: int) -> str:
    if index >= len(text):
        return text
    if index <= 0:
        return ""
    prefix = text[:index]
    if not text[index].isspace():
        last_space = prefix.rfind(" ")
        if last_space > 0:
            prefix = prefix[:last_space]
    return prefix.strip()


class RimeWebsocketSynthesizer(BaseSynthesizer[RimeSynthesizerConfig], InputStreamingSynthesizer):
    """Coda websocket synthesizer.

    Coda addresses every operation to a contextId, which lets one connection carry both
    modes this pipeline needs: a context per message for the pre-call cache, where the
    audio has to arrive as a discrete blob, and a context per turn for the live
    conversation, where prosody is supposed to carry across sentences.
    """

    def __init__(self, synthesizer_config: RimeSynthesizerConfig):
        super().__init__(synthesizer_config)

        self.speaker = synthesizer_config.speaker
        self.speed_alpha = synthesizer_config.speed_alpha
        self.sampling_rate = synthesizer_config.sampling_rate
        self.api_key = f"Bearer {getenv('RIME_API_KEY')}"

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.listener: Optional[asyncio.Task] = None
        self.connect_lock = asyncio.Lock()
        self.context_queues: dict[str, asyncio.Queue] = {}

        self.turn_context_id: Optional[str] = None
        self.turn_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.turn_text_buffer = ""
        self.turn_text_sent = ""
        self.turn_audio_bytes = 0

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: RimeSynthesizerConfig):
        return RimeSynthesizer.get_voice_identifier(synthesizer_config)

    # --- connection ------------------------------------------------------------------

    async def establish_connection(self):
        async with self.connect_lock:
            if (
                self.websocket is not None
                and self.listener is not None
                and not self.listener.done()
            ):
                return
            self.websocket = await websockets.connect(
                RIME_CODA_WS_URL,
                extra_headers={"Authorization": self.api_key},
                subprotocols=[RIME_CODA_SUBPROTOCOL],
            )
            ready = json.loads(await self.websocket.recv())
            if "ready" not in ready:
                raise RimeWebsocketError(f"Expected a ready frame from Rime, received {ready}")
            self.listener = asyncio_create_task(self.listen())

    async def listen(self):
        try:
            async for raw in self.websocket:
                frame = json.loads(raw)
                queue = self.context_queues.get(frame.get("contextId"))
                if queue is None:
                    continue
                if "audio" in frame:
                    audio = base64.b64decode(frame["audio"])
                    if frame.get("contextId") == self.turn_context_id:
                        self.turn_audio_bytes += len(audio)
                    queue.put_nowait(audio)
                elif "error" in frame:
                    logger.error(f"Rime websocket returned an error: {frame['error']}")
                    queue.put_nowait(None)
                elif "done" in frame or "cancelled" in frame:
                    queue.put_nowait(None)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Rime websocket listener terminated: {e}")
            for queue in self.context_queues.values():
                queue.put_nowait(None)

    async def send_operation(self, payload: dict):
        assert self.websocket is not None, "Rime websocket is not connected"
        await self.websocket.send(json.dumps(payload))

    def get_audio_parameters(self) -> dict:
        speed_alpha = self.speed_alpha if self.speed_alpha else RIME_DEFAULT_SPEED_ALPHA
        # Coda scales duration rather than rate, so it is the reciprocal of speedAlpha.
        return {
            "audioFormat": "audio/PCMU",
            "samplingRate": self.sampling_rate,
            "timeScaleFactor": 1 / speed_alpha,
        }

    async def start_context(self, context_id: str, text: str = ""):
        await self.send_operation(
            {
                "contextId": context_id,
                "start": {
                    "speaker": self.speaker,
                    "language": "en",
                    "text": text,
                    "audioParameters": self.get_audio_parameters(),
                },
            }
        )

    async def drain_context(self, context_id: str, queue: asyncio.Queue):
        try:
            async for chunk_result in self.chunk_result_generator_from_queue(queue):
                yield chunk_result
        finally:
            self.context_queues.pop(context_id, None)

    # --- cached path: one context per message -----------------------------------------

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        await self.establish_connection()
        self.total_chars += len(message.text)

        context_id = str(uuid.uuid4())
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.context_queues[context_id] = queue

        await self.start_context(context_id)
        await self.send_operation({"contextId": context_id, "text": message.text})
        await self.send_operation({"contextId": context_id, "end": {}})

        return SynthesisResult(
            self.drain_context(context_id, queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    # --- live path: one context per turn ----------------------------------------------

    def ready_synthesizer(self, chunk_size: int):
        asyncio_create_task(self.establish_connection())

    async def send_token_to_synthesizer(self, message: LLMToken, chunk_size: int):
        await self.establish_connection()

        if self.turn_context_id is None:
            self.turn_context_id = str(uuid.uuid4())
            self.turn_queue = asyncio.Queue()
            self.turn_text_buffer = ""
            self.turn_text_sent = ""
            self.turn_audio_bytes = 0
            self.context_queues[self.turn_context_id] = self.turn_queue
            await self.start_context(self.turn_context_id)

        self.turn_text_buffer += message.text
        if self.turn_text_buffer.rstrip().endswith(SENTENCE_ENDINGS):
            await self.flush_turn_buffer()

    async def flush_turn_buffer(self):
        text = self.turn_text_buffer
        self.turn_text_buffer = ""
        if not text.strip():
            return
        text = text if text.endswith(" ") else text + " "
        self.turn_text_sent += text
        self.total_chars += len(text)
        await self.send_operation({"contextId": self.turn_context_id, "text": text})

    def get_current_utterance_synthesis_result(self):
        return SynthesisResult(
            self.chunk_result_generator_from_queue(self.turn_queue),
            lambda seconds: self.get_current_message_so_far(seconds),
        )

    def get_current_message_so_far(self, seconds: Optional[float]) -> str:
        text = (self.turn_text_sent + self.turn_text_buffer).strip()
        if seconds is None or not text:
            return text
        # Coda carries no word timings yet, so the elapsed text is derived from the audio
        # actually received rather than from an assumed words-per-minute rate.
        if not self.turn_audio_bytes:
            return BaseSynthesizer.get_message_cutoff_from_voice_speed(
                BaseMessage(text=text), seconds, 150
            )
        seconds_per_char = (self.turn_audio_bytes / self.sampling_rate) / len(text)
        return snap_to_word_boundary(text, int(seconds / seconds_per_char))

    async def handle_end_of_turn(self):
        if self.turn_context_id is None:
            return
        await self.flush_turn_buffer()
        await self.send_operation({"contextId": self.turn_context_id, "end": {}})
        self.turn_context_id = None

    # --- teardown ---------------------------------------------------------------------

    async def handle_interrupt(self):
        context_id = self.turn_context_id
        self.turn_context_id = None
        if context_id is None:
            return
        queue = self.context_queues.pop(context_id, None)
        if queue is not None:
            queue.put_nowait(None)
        try:
            await self.send_operation({"contextId": context_id, "cancel": {}})
        except Exception as e:
            logger.error(f"Failed to cancel Rime context: {e}")

    async def cancel_websocket_tasks(self):
        await self.handle_interrupt()

    async def tear_down(self):
        await self.cancel_websocket_tasks()
        if self.listener is not None:
            self.listener.cancel()
            self.listener = None
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        self.context_queues.clear()
        await super().tear_down()
