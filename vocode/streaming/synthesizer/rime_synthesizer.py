import asyncio
import audioop
import base64
import io
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import websockets
from loguru import logger

from app import call_id as ctx_call_id

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    RIME_DEFAULT_REDUCE_LATENCY,
    RIME_DEFAULT_SPEED_ALPHA,
    RimeSynthesizerConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.utils.create_task import asyncio_create_task

# TODO: [OSS] Remove call to internal library with Synthesizers refactor

# https://rime.ai/docs/quickstart

WAV_HEADER_LENGTH = 44

RIME_WS_URL = "wss://api.rime.ai/coda/ws"
RIME_WS_SUBPROTOCOL = "rime.v1.json"
RIME_WS_AUDIO_FORMAT = "audio/PCMU"


class RimeError(Exception):
    pass


class RimeSynthesizer(BaseSynthesizer[RimeSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: RimeSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.base_url = synthesizer_config.base_url
        self.model_id = synthesizer_config.model_id
        speaker, _, suffix = synthesizer_config.speaker.rpartition("-")
        self.cache_speed_alpha = None
        self.use_websocket = suffix == "ws"
        if self.use_websocket:
            self.speaker = speaker
        else:
            try:
                self.cache_speed_alpha = float(suffix)
                self.speaker = speaker
            except ValueError:
                self.speaker = synthesizer_config.speaker
        self._ws = None
        self._ws_context_id = None
        self._ws_queue = None
        self._ws_reader = None
        self._ws_chunk_size = None
        self.speed_alpha = synthesizer_config.speed_alpha
        self.sampling_rate = synthesizer_config.sampling_rate
        self.reduce_latency = synthesizer_config.reduce_latency
        self.api_key = f"Bearer {getenv('RIME_API_KEY')}"

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: RimeSynthesizerConfig):
        return ":".join(
            (
                "rime",
                synthesizer_config.speaker,
                str(synthesizer_config.speed_alpha),
                synthesizer_config.audio_encoding,
            )
        )

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        use_pcm = self.model_id in ("mistv3", "coda")
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            **({"Accept": "audio/PCMU"} if use_pcm else {}),
        }

        body = self.get_request_body(message.text)

        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task(
            self.get_chunks(headers, body, chunk_size, chunk_queue),
        )

        return SynthesisResult(
            self.chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    @staticmethod
    async def _chunk_generator(output_bytes, chunk_size):
        for i in range(0, len(output_bytes), chunk_size):
            if i + chunk_size > len(output_bytes):
                yield SynthesisResult.ChunkResult(output_bytes[i:], True)
            else:
                yield SynthesisResult.ChunkResult(output_bytes[i : i + chunk_size], False)

    async def get_chunks(
        self,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue[Optional[bytes]],
    ):
        call_id = str(ctx_call_id.value) if ctx_call_id.value else None
        logger.info(
            "Rime request: "
            + json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "url": self.base_url,
                    "body": body,
                    "call_id": call_id,
                },
                ensure_ascii=False,
            )
        )
        started_at = time.monotonic()
        total_bytes = 0
        first_chunk_at = None
        try:
            async_client = self.async_requestor.get_client()
            stream = await async_client.send(
                async_client.build_request(
                    "POST",
                    self.base_url,
                    headers=headers,
                    json=body,
                ),
                stream=True,
            )
            if not stream.is_success:
                error = await stream.aread()
                text = error.decode("utf-8", "replace")
                try:
                    detail = json.loads(text)
                except ValueError:
                    detail = text.strip()
                logger.error(
                    "Rime response error: "
                    + json.dumps(
                        {
                            "status": stream.status_code,
                            "content_type": stream.headers.get("content-type", ""),
                            "error": detail,
                            "call_id": call_id,
                        },
                        ensure_ascii=False,
                    )
                )
                raise RimeError(
                    f"Rime API returned {stream.status_code} status code with the following details: {text}"
                )
            async for chunk in stream.aiter_bytes(chunk_size):
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                total_bytes += len(chunk)
                chunk_queue.put_nowait(chunk)
            logger.info(
                "Rime response: "
                + json.dumps(
                    {
                        "status": stream.status_code,
                        "headers": dict(stream.headers),
                        "total_bytes": total_bytes,
                        "ttfb_ms": (
                            round((first_chunk_at - started_at) * 1000) if first_chunk_at else None
                        ),
                        "ttlb_ms": round((time.monotonic() - started_at) * 1000),
                        "call_id": call_id,
                    },
                    ensure_ascii=False,
                )
            )
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel

    async def _ws_connect(self):
        if self._ws is not None and not self._ws.closed:
            return self._ws
        self._ws = await websockets.connect(
            RIME_WS_URL,
            subprotocols=[RIME_WS_SUBPROTOCOL],
            extra_headers={"Authorization": self.api_key},
        )
        await self._ws.recv()
        return self._ws

    async def _ws_read(self, context_id: str, chunk_queue: asyncio.Queue, chunk_size: int):
        started_at = time.monotonic()
        total_bytes = 0
        first_chunk_at = None
        buffer = b""
        try:
            while True:
                message = json.loads(await self._ws.recv())
                if message.get("contextId") != context_id:
                    continue
                if "audio" in message:
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    decoded = base64.b64decode(message["audio"])
                    total_bytes += len(decoded)
                    buffer += decoded
                    while len(buffer) >= chunk_size:
                        chunk_queue.put_nowait(buffer[:chunk_size])
                        buffer = buffer[chunk_size:]
                    continue
                if "error" in message:
                    logger.error(
                        "Rime response error: "
                        + json.dumps({"context_id": context_id, "error": message["error"]},
                                     ensure_ascii=False)
                    )
                    break
                if "done" in message or "cancelled" in message:
                    break
            if buffer:
                chunk_queue.put_nowait(buffer)
            logger.info(
                "Rime response: "
                + json.dumps(
                    {
                        "context_id": context_id,
                        "total_bytes": total_bytes,
                        "ttfb_ms": (
                            round((first_chunk_at - started_at) * 1000) if first_chunk_at else None
                        ),
                        "ttlb_ms": round((time.monotonic() - started_at) * 1000),
                        "call_id": str(ctx_call_id.value) if ctx_call_id.value else None,
                    },
                    ensure_ascii=False,
                )
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Rime websocket reader failed: {e}")
        finally:
            chunk_queue.put_nowait(None)

    async def _ws_close_context(self, operation: str):
        if self._ws_context_id is None:
            return
        try:
            await self._ws.send(
                json.dumps({"contextId": self._ws_context_id, operation: {}})
            )
        except Exception as e:
            logger.error(f"Rime websocket {operation} failed: {e}")
        if operation == "cancel" and self._ws_reader is not None:
            self._ws_reader.cancel()
        self._ws_context_id = None
        self._ws_queue = None
        self._ws_reader = None

    async def handle_end_of_turn(self):
        await self._ws_close_context("end")

    async def handle_interrupt(self):
        await self._ws_close_context("cancel")

    @staticmethod
    async def _no_audio():
        return
        yield

    async def create_speech_ws(
        self, message: BaseMessage, chunk_size: int, is_first_text_chunk: bool
    ) -> SynthesisResult:
        """One context per agent turn. The first sentence carries the turn's audio;
        later sentences are sent into the same context and return no audio of their own."""
        body = self.get_request_body(message.text)
        self.total_chars += len(message.text)
        await self._ws_connect()

        if is_first_text_chunk or self._ws_context_id is None:
            if self._ws_context_id is not None:
                await self._ws_close_context("cancel")
            self._ws_context_id = str(uuid.uuid4())
            self._ws_queue = asyncio.Queue()
            self._ws_chunk_size = chunk_size
            start = {
                "speaker": self.speaker,
                "language": "en",
                "text": "",
                "audioParameters": {
                    "audioFormat": RIME_WS_AUDIO_FORMAT,
                    "samplingRate": self.sampling_rate,
                    "timeScaleFactor": 1 / (body.get("speedAlpha") or 1),
                },
            }
            await self._ws.send(
                json.dumps({"contextId": self._ws_context_id, "start": start})
            )
            self._ws_reader = asyncio_create_task(
                self._ws_read(self._ws_context_id, self._ws_queue, chunk_size)
            )
            queue = self._ws_queue
        else:
            queue = None

        logger.info(
            "Rime request: "
            + json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "url": RIME_WS_URL,
                    "context_id": self._ws_context_id,
                    "first": is_first_text_chunk,
                    "text": body["text"],
                    "call_id": str(ctx_call_id.value) if ctx_call_id.value else None,
                },
                ensure_ascii=False,
            )
        )
        await self._ws.send(
            json.dumps({"contextId": self._ws_context_id, "text": body["text"]})
        )

        if queue is None:
            return SynthesisResult(self._no_audio(), lambda seconds: message.text)
        return SynthesisResult(
            self.chunk_result_generator_from_queue(queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    def get_request_body(self, text):
        speed_alpha = self.speed_alpha if self.speed_alpha else RIME_DEFAULT_SPEED_ALPHA
        reduce_latency = self.reduce_latency if self.reduce_latency else RIME_DEFAULT_REDUCE_LATENCY

        body = {
            "text": text,
            "speaker": self.speaker,
            "samplingRate": self.sampling_rate,
            "speedAlpha": speed_alpha,
            "reduceLatency": reduce_latency,
        }

        if self.model_id:
            body["modelId"] = self.model_id

        return body
