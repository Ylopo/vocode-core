import asyncio
import base64
import json
import re
import time
import uuid
from datetime import datetime, timezone
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

MODE_CACHED = "cached"
MODE_LIVE = "live"

SENTENCE_BOUNDARY = re.compile(r"([.!?])(\s+)")


class RimeWebsocketError(Exception):
    pass


def find_release_point(text: str) -> int:
    """Index just past a releasable sentence boundary, or -1 if there is not one yet.

    Each text frame is a prosodic unit to Coda, so the two ways of being wrong do not
    cost the same. Releasing at a boundary that is not one forces a break inside a name
    or a time; failing to release at a real one only means Rime receives a larger frame,
    which is the direction this whole path exists to move in. So a period closes a
    sentence only where nothing about it looks abbreviated, and anything doubtful is
    held for the next token - or for the end of the turn, which always flushes.
    """
    for match in SENTENCE_BOUNDARY.finditer(text):
        if match.group(1) == ".":
            preceding = text[: match.start()].split()
            token = preceding[-1] if preceding else ""
            following = text[match.end() : match.end() + 1]
            if (
                len(token) < 3  # an initial or a short title: L. St. Mr. Dr.
                or "." in token  # an internal period: p.m. J.R. U.S.
                or not token[-1].isalpha()  # a numbered item or a figure: 1. 9375.
                or not following  # the next word has not arrived to judge by
                or not following.isupper()  # an abbreviation mid-sentence: a.m. and
            ):
                continue
        return match.end()
    return -1


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self.connection_id: Optional[str] = None
        self.context_queues: dict[str, asyncio.Queue] = {}
        self.context_stats: dict[str, dict] = {}

        self.turn_context_id: Optional[str] = None
        self.turn_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.turn_text_buffer = ""
        self.turn_text_sent = ""
        self.turn_audio_bytes = 0

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: RimeSynthesizerConfig):
        return RimeSynthesizer.get_voice_identifier(synthesizer_config)

    # --- logging ----------------------------------------------------------------------
    #
    # Every frame exchanged with Rime is logged verbatim except audio, which is counted
    # rather than printed: a single turn carries a few hundred audio frames, and logging
    # each one would cost more than the synthesis it is describing. What audio would have
    # told us - arrival time, count, total size and framing - is captured on the context
    # summary instead.

    def log_event(self, event: str, payload: dict):
        logger.info(f"Rime {event}: " + json.dumps(payload, ensure_ascii=False, default=str))

    def context_elapsed_ms(self, context_id: Optional[str]) -> Optional[int]:
        stats = self.context_stats.get(context_id) if context_id else None
        if stats is None:
            return None
        return round((time.monotonic() - stats["started_at"]) * 1000)

    def log_context_result(self, context_id: str, outcome: str):
        stats = self.context_stats.pop(context_id, None)
        if stats is None:
            return
        sizes = stats["frame_sizes"]
        first_audio_at = stats["first_audio_at"]
        self.log_event(
            "context",
            {
                "at": now_iso(),
                "connection_id": self.connection_id,
                "context_id": context_id,
                "mode": stats["mode"],
                "outcome": outcome,
                "speaker": self.speaker,
                "audio_parameters": stats["audio_parameters"],
                "text": stats["text"],
                "text_frames": stats["text_frames"],
                "audio_frames": stats["frames"],
                "audio_bytes": stats["bytes"],
                "audio_seconds": round(stats["bytes"] / self.sampling_rate, 3),
                "frame_bytes_min": min(sizes) if sizes else None,
                "frame_bytes_max": max(sizes) if sizes else None,
                "ttfb_ms": (
                    round((first_audio_at - stats["started_at"]) * 1000) if first_audio_at else None
                ),
                "ttlb_ms": round((time.monotonic() - stats["started_at"]) * 1000),
            },
        )

    # --- connection ------------------------------------------------------------------

    async def establish_connection(self):
        async with self.connect_lock:
            if (
                self.websocket is not None
                and self.listener is not None
                and not self.listener.done()
            ):
                return
            opened_at = time.monotonic()
            self.connection_id = str(uuid.uuid4())
            self.websocket = await websockets.connect(
                RIME_CODA_WS_URL,
                extra_headers={"Authorization": self.api_key},
                subprotocols=[RIME_CODA_SUBPROTOCOL],
            )
            ready = json.loads(await self.websocket.recv())
            if "ready" not in ready:
                raise RimeWebsocketError(f"Expected a ready frame from Rime, received {ready}")
            self.listener = asyncio_create_task(self.listen())
            self.log_event(
                "connected",
                {
                    "at": now_iso(),
                    "connection_id": self.connection_id,
                    "url": RIME_CODA_WS_URL,
                    "subprotocol": RIME_CODA_SUBPROTOCOL,
                    "handshake_ms": round((time.monotonic() - opened_at) * 1000),
                    "frame": ready,
                },
            )

    def register_context(
        self, context_id: str, queue: asyncio.Queue, mode: str, chunk_size: int = 0
    ):
        self.context_queues[context_id] = queue
        self.context_stats[context_id] = {
            "started_at": time.monotonic(),
            "first_audio_at": None,
            "frames": 0,
            "bytes": 0,
            "frame_sizes": [],
            "text_frames": 0,
            "text": "",
            "mode": mode,
            "chunk_size": chunk_size,
            "buffer": bytearray(),
            "audio_parameters": None,
        }

    async def listen(self):
        try:
            async for raw in self.websocket:
                frame = json.loads(raw)
                context_id = frame.get("contextId")
                stats = self.context_stats.get(context_id)
                queue = self.context_queues.get(context_id)

                if "audio" in frame:
                    audio = base64.b64decode(frame["audio"])
                    if stats is not None:
                        if stats["first_audio_at"] is None:
                            stats["first_audio_at"] = time.monotonic()
                            self.log_event(
                                "<< first audio",
                                {
                                    "at": now_iso(),
                                    "context_id": context_id,
                                    "mode": stats["mode"],
                                    "bytes": len(audio),
                                    "ttfb_ms": self.context_elapsed_ms(context_id),
                                },
                            )
                        stats["frames"] += 1
                        stats["bytes"] += len(audio)
                        stats["frame_sizes"].append(len(audio))
                    if queue is None:
                        continue
                    if stats is None or stats["mode"] == MODE_CACHED:
                        # Cached audio is accumulated into one blob and re-chunked when it
                        # is played back, so it can be queued exactly as it arrives.
                        queue.put_nowait(audio)
                        continue
                    self.turn_audio_bytes += len(audio)
                    # Live audio goes straight to the output device, which expects fixed
                    # size frames; Rime's are whatever the model happened to emit.
                    buffer, chunk_size = stats["buffer"], stats["chunk_size"]
                    buffer.extend(audio)
                    while chunk_size > 0 and len(buffer) >= chunk_size:
                        queue.put_nowait(bytes(buffer[:chunk_size]))
                        del buffer[:chunk_size]
                    continue

                # Everything that is not audio is logged exactly as Rime sent it.
                self.log_event(
                    "<<",
                    {
                        "at": now_iso(),
                        "connection_id": self.connection_id,
                        "context_id": context_id,
                        "elapsed_ms": self.context_elapsed_ms(context_id),
                        "frame": frame,
                    },
                )
                if queue is None:
                    continue
                if "error" in frame:
                    self.log_context_result(context_id, "error")
                    queue.put_nowait(None)
                elif "done" in frame or "cancelled" in frame:
                    if stats is not None and stats["buffer"]:
                        queue.put_nowait(bytes(stats["buffer"]))
                        stats["buffer"].clear()
                    self.log_context_result(
                        context_id, "cancelled" if "cancelled" in frame else "done"
                    )
                    queue.put_nowait(None)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Rime websocket listener terminated: {e}")
            for queue in self.context_queues.values():
                queue.put_nowait(None)

    async def send_operation(self, payload: dict):
        assert self.websocket is not None, "Rime websocket is not connected"
        context_id = payload.get("contextId")
        stats = self.context_stats.get(context_id) if context_id else None
        if stats is not None:
            if "text" in payload:
                stats["text_frames"] += 1
                stats["text"] += payload["text"]
            elif "start" in payload:
                stats["audio_parameters"] = payload["start"].get("audioParameters")
        self.log_event(
            ">>",
            {
                "at": now_iso(),
                "connection_id": self.connection_id,
                "context_id": context_id,
                "mode": stats["mode"] if stats else None,
                "elapsed_ms": self.context_elapsed_ms(context_id),
                "payload": payload,
            },
        )
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
            self.context_stats.pop(context_id, None)

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
        self.register_context(context_id, queue, mode=MODE_CACHED)

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
            self.register_context(
                self.turn_context_id, self.turn_queue, mode=MODE_LIVE, chunk_size=chunk_size
            )
            await self.start_context(self.turn_context_id)

        self.turn_text_buffer += message.text
        while True:
            release_at = find_release_point(self.turn_text_buffer)
            if release_at < 0:
                break
            await self.flush_turn_buffer(release_at)

    async def flush_turn_buffer(self, release_at: Optional[int] = None):
        # Sliced out of the raw buffer, never reassembled from stripped pieces, so the
        # whitespace either side of a boundary survives and words are not run together.
        if release_at is None:
            text, self.turn_text_buffer = self.turn_text_buffer, ""
        else:
            text = self.turn_text_buffer[:release_at]
            self.turn_text_buffer = self.turn_text_buffer[release_at:]
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
        self.log_context_result(context_id, "interrupted")

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
        self.context_stats.clear()
        await super().tear_down()
