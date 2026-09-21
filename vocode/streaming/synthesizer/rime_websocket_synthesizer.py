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
RIME_LOGGED_HEADERS = {"Authorization": "Bearer ***REDACTED***"}

# A context is either a single self-contained message or a whole agent turn. It says
# nothing about where the audio ends up: a one-shot context feeds the pre-call cache,
# but mid-call it feeds the output device directly.
MODE_ONESHOT = "oneshot"
MODE_STREAM = "stream"

SENTENCE_BOUNDARY = re.compile(r"([.!?])(\s+)")
WHITESPACE_RUN = re.compile(r"\s+")


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


def short_id(identifier: Optional[str]) -> str:
    return identifier.split("-")[0] if identifier else "-"


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
        self.connection_opened_at: Optional[float] = None
        self.connection_opened_iso: Optional[str] = None
        self.contexts_served = 0
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
    # One JSON block per context, not a line per frame. The exchange is accumulated as it
    # happens and emitted once when the context closes, so a whole message reads as a
    # single object rather than as a dozen lines interleaved with every other context on
    # the connection. Audio frames are counted rather than recorded - a turn carries a few
    # hundred - and what they would have told us, arrival time, count, total size and
    # framing, is on the summary instead.

    def log_block(self, kind: str, summary: str, block: dict):
        # Bound rather than formatted into the message: the json sink serializes `extra`
        # as nested json, so the block arrives in the log as an object that expands and
        # can be queried by field, instead of one long escaped string. The summary line
        # carries the same story in one line, which is all a pretty sink would show.
        logger.bind(rime={"kind": kind, **block}).info(f"Rime {kind} {summary}")

    def connection_block(self) -> dict:
        return {
            "id": self.connection_id,
            "opened_at": self.connection_opened_iso,
            "url": RIME_CODA_WS_URL,
            "subprotocol": RIME_CODA_SUBPROTOCOL,
            "headers": RIME_LOGGED_HEADERS,
        }

    def record(self, stats: Optional[dict], direction: str, frame: dict):
        """Append every operation in a frame to its context's exchange, audio excepted."""
        if stats is None:
            return
        at_ms = round((time.monotonic() - stats["started_at"]) * 1000)
        for op, payload in frame.items():
            if op != "contextId":
                stats["exchange"].append(
                    {"at_ms": at_ms, "dir": direction, "op": op, "payload": payload}
                )

    def log_orphan(self, direction: str, frame: dict):
        # A frame for a context that is already closed, or was never registered. It has
        # no block to belong to, and dropping it would hide the fault that produced it.
        context_id = frame.get("contextId")
        self.log_block(
            "orphan",
            f"{direction} {short_id(context_id)}",
            {
                "connection": self.connection_block(),
                "context_id": context_id,
                "dir": direction,
                "frame": frame,
            },
        )

    def log_context_result(self, context_id: str, outcome: str):
        stats = self.context_stats.pop(context_id, None)
        if stats is None:
            return
        sizes = stats["frame_sizes"]
        first_audio_at = stats["first_audio_at"]
        seconds = round(stats["bytes"] / self.sampling_rate, 3)
        ttfb_ms = round((first_audio_at - stats["started_at"]) * 1000) if first_audio_at else None
        ttlb_ms = round((time.monotonic() - stats["started_at"]) * 1000)
        self.log_block(
            "context",
            f"{short_id(context_id)} | {stats['mode']} | {outcome} | {seconds}s audio | "
            f"ttfb {f'{ttfb_ms}ms' if ttfb_ms is not None else 'none'} | ttlb {ttlb_ms}ms",
            {
                "context_id": context_id,
                "mode": stats["mode"],
                "outcome": outcome,
                "started_at": stats["started_iso"],
                "connection": self.connection_block(),
                "speaker": self.speaker,
                "audio_parameters": stats["audio_parameters"],
                "text": stats["text"],
                "text_frames": stats["text_frames"],
                "audio": {
                    "frames": stats["frames"],
                    "bytes": stats["bytes"],
                    "seconds": seconds,
                    "frame_bytes": {
                        "min": min(sizes) if sizes else None,
                        "max": max(sizes) if sizes else None,
                    },
                },
                "timing": {"ttfb_ms": ttfb_ms, "ttlb_ms": ttlb_ms},
                "exchange": stats["exchange"],
            },
        )

    def log_disconnected(self, reason: str, error: Optional[str] = None):
        open_ms = self.connection_open_ms()
        in_flight = len(self.context_queues)
        self.log_block(
            "disconnected",
            f"{short_id(self.connection_id)} | {reason} | open {open_ms}ms | "
            f"{self.contexts_served} contexts | {in_flight} in flight",
            {
                "reason": reason,
                "error": error,
                "connection": self.connection_block(),
                "open_ms": open_ms,
                "contexts_served": self.contexts_served,
                "contexts_in_flight": in_flight,
            },
        )
        self.log_abandoned_contexts()

    def log_abandoned_contexts(self):
        # A context only emits its block when it closes, so one that never closed would
        # otherwise leave no trace at all - which is exactly the case worth seeing.
        for context_id in list(self.context_stats):
            self.log_context_result(context_id, "abandoned")

    # --- connection ------------------------------------------------------------------

    async def establish_connection(self):
        async with self.connect_lock:
            if (
                self.websocket is not None
                and self.listener is not None
                and not self.listener.done()
            ):
                return
            self.connection_opened_at = opened_at = time.monotonic()
            self.connection_opened_iso = now_iso()
            self.connection_id = str(uuid.uuid4())
            self.contexts_served = 0
            self.websocket = await websockets.connect(
                RIME_CODA_WS_URL,
                extra_headers={"Authorization": self.api_key},
                subprotocols=[RIME_CODA_SUBPROTOCOL],
            )
            ready = json.loads(await self.websocket.recv())
            if "ready" not in ready:
                raise RimeWebsocketError(f"Expected a ready frame from Rime, received {ready}")
            self.listener = asyncio_create_task(self.listen())
            handshake_ms = round((time.monotonic() - opened_at) * 1000)
            self.log_block(
                "connected",
                f"{short_id(self.connection_id)} | handshake {handshake_ms}ms",
                {
                    "connection": self.connection_block(),
                    "handshake_ms": handshake_ms,
                    "frame": ready,
                },
            )

    def register_context(
        self, context_id: str, queue: asyncio.Queue, mode: str, chunk_size: int = 0
    ):
        self.contexts_served += 1
        self.context_queues[context_id] = queue
        self.context_stats[context_id] = {
            "started_at": time.monotonic(),
            "started_iso": now_iso(),
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
            "exchange": [],
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
                            self.record(stats, "<<", {"audio": {"first frame bytes": len(audio)}})
                        stats["frames"] += 1
                        stats["bytes"] += len(audio)
                        stats["frame_sizes"].append(len(audio))
                    if queue is None:
                        continue
                    if stats is None:
                        queue.put_nowait(audio)
                        continue
                    if stats["mode"] == MODE_STREAM:
                        self.turn_audio_bytes += len(audio)
                    # Rime emits frames of whatever size the model produced. Every context
                    # is re-chunked, because a one-shot context is only sometimes drained
                    # into a blob - mid-call it feeds the output device directly, and that
                    # expects fixed size frames.
                    buffer, chunk_size = stats["buffer"], stats["chunk_size"]
                    buffer.extend(audio)
                    while chunk_size > 0 and len(buffer) >= chunk_size:
                        queue.put_nowait(bytes(buffer[:chunk_size]))
                        del buffer[:chunk_size]
                    continue

                # Everything that is not audio is recorded exactly as Rime sent it.
                if stats is None:
                    self.log_orphan("<<", frame)
                else:
                    self.record(stats, "<<", frame)
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
                    # A finished context is dead: only the one-shot path cleaned up after
                    # itself, so a turn's queue would otherwise be held for the whole call.
                    self.context_queues.pop(context_id, None)
                    queue.put_nowait(None)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_disconnected("listener terminated", str(e))
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
            self.record(stats, ">>", payload)
        else:
            self.log_orphan(">>", payload)
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

    # --- one-shot: a context per message ----------------------------------------------

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
        self.register_context(context_id, queue, mode=MODE_ONESHOT, chunk_size=chunk_size)

        await self.start_context(context_id)
        await self.send_operation({"contextId": context_id, "text": message.text})
        await self.send_operation({"contextId": context_id, "end": {}})

        return SynthesisResult(
            self.drain_context(context_id, queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    # --- streaming: one context per turn ----------------------------------------------

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
                self.turn_context_id, self.turn_queue, mode=MODE_STREAM, chunk_size=chunk_size
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
        # The agent's token collator emits a space of its own whenever a token opens with
        # punctuation, so the stream arrives with doubled spaces the HTTP path never has.
        # They reach Rime and the transcript alike, so they are collapsed here.
        text = WHITESPACE_RUN.sub(" ", text).lstrip()
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
        # Only text Rime was actually sent can have been heard. The unflushed tail never
        # reached the model, so it is no part of the turn as far as the caller - or the
        # agent's own history, which this string becomes - is concerned. Collapsed to
        # single spaces so the transcript reads as it does on the HTTP path.
        text = WHITESPACE_RUN.sub(" ", self.turn_text_sent).strip()
        if seconds is None or not text:
            return text
        # Mulaw arrives at a fixed byte per sample, so the audio that has come back is
        # its own clock: its length in seconds is the length of the speech. How far
        # playback got through it is the proportion of the text that was heard, and no
        # assumed speaking rate comes into it.
        audio_seconds = self.turn_audio_bytes / self.sampling_rate
        if not audio_seconds:
            return ""
        heard = min(seconds / audio_seconds, 1.0)
        return snap_to_word_boundary(text, int(len(text) * heard))

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

    def connection_open_ms(self) -> Optional[int]:
        if self.connection_opened_at is None:
            return None
        return round((time.monotonic() - self.connection_opened_at) * 1000)

    async def tear_down(self):
        await self.cancel_websocket_tasks()
        if self.listener is not None:
            self.listener.cancel()
            self.listener = None
        if self.websocket is not None:
            self.log_disconnected("torn down")
            await self.websocket.close()
            self.websocket = None
        self.connection_opened_at = None
        self.connection_opened_iso = None
        self.context_queues.clear()
        self.context_stats.clear()
        await super().tear_down()
