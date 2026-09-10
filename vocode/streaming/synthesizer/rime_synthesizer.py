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

from vocode import conversation_id as ctx_conversation_id
from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    RIME_DEFAULT_REDUCE_LATENCY,
    RIME_DEFAULT_SPEED_ALPHA,
    RimeSynthesizerConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult, strip_non_speech
from vocode.streaming.utils.create_task import asyncio_create_task

# TODO: [OSS] Remove call to internal library with Synthesizers refactor

# https://rime.ai/docs/quickstart

WAV_HEADER_LENGTH = 44

RIME_WS_URL = "wss://api.rime.ai/coda/ws"
RIME_WS_SUBPROTOCOL = "rime.v1.json"
RIME_WS_AUDIO_PARAMETERS = {"audioFormat": "audio/PCMU", "samplingRate": 8000}


class RimeError(Exception):
    pass


def _rime_log_ids(synthesizer):
    conversation = getattr(synthesizer, "streaming_conversation", None)
    return {
        "conversation_id": ctx_conversation_id.value
        or (conversation.id if conversation else None),
        "call_id": str(ctx_call_id.value) if ctx_call_id.value else None,
    }


class RimeSynthesizer(BaseSynthesizer[RimeSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: RimeSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.base_url = synthesizer_config.base_url
        self.model_id = synthesizer_config.model_id
        self.use_websocket = synthesizer_config.speaker.endswith("-ws")
        self.speaker = synthesizer_config.speaker.removesuffix("-ws")
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
        ids = _rime_log_ids(self)
        logger.info(
            "Rime request: "
            + json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "transport": "http",
                    "url": self.base_url,
                    "body": body,
                    **ids,
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
                            "transport": "http",
                            "status": stream.status_code,
                            "content_type": stream.headers.get("content-type", ""),
                            "error": detail,
                            **ids,
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
                        "transport": "http",
                        "status": stream.status_code,
                        "headers": dict(stream.headers),
                        "total_bytes": total_bytes,
                        "ttfb_ms": (
                            round((first_chunk_at - started_at) * 1000) if first_chunk_at else None
                        ),
                        "ttlb_ms": round((time.monotonic() - started_at) * 1000),
                        **ids,
                    },
                    ensure_ascii=False,
                )
            )
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel

    async def get_ws_chunks(self, body: dict, chunk_size: int, chunk_queue: asyncio.Queue):
        context_id = str(uuid.uuid4())
        ids = _rime_log_ids(self)
        start = {
            "speaker": body["speaker"],
            "language": "en",
            "text": "",
            "audioParameters": {
                **RIME_WS_AUDIO_PARAMETERS,
                "timeScaleFactor": 1 / (body.get("speedAlpha") or 1),
            },
        }
        logger.info(
            "Rime request: "
            + json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "transport": "websocket",
                    "url": RIME_WS_URL,
                    "context_id": context_id,
                    "start": start,
                    "text": body["text"],
                    **ids,
                },
                ensure_ascii=False,
            )
        )
        started_at = time.monotonic()
        total_bytes = 0
        frames = 0
        first_chunk_at = None
        request_id = ""
        buffer = b""
        try:
            async with websockets.connect(
                RIME_WS_URL,
                subprotocols=[RIME_WS_SUBPROTOCOL],
                extra_headers={"Authorization": self.api_key},
            ) as ws:
                await ws.recv()
                await ws.send(json.dumps({"contextId": context_id, "start": start}))
                await ws.send(json.dumps({"contextId": context_id, "text": body["text"]}))
                await ws.send(json.dumps({"contextId": context_id, "end": {}}))
                while True:
                    message = json.loads(await ws.recv())
                    if "audio" in message:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        decoded = base64.b64decode(message["audio"])
                        total_bytes += len(decoded)
                        frames += 1
                        buffer += decoded
                        while len(buffer) >= chunk_size:
                            chunk_queue.put_nowait(buffer[:chunk_size])
                            buffer = buffer[chunk_size:]
                        continue
                    if "started" in message:
                        request_id = message["started"].get("requestId", "")
                    if "error" in message:
                        logger.error(
                            "Rime response error: "
                            + json.dumps(
                                {
                                    "transport": "websocket",
                                    "context_id": context_id,
                                    "error": message["error"],
                                    **ids,
                                },
                                ensure_ascii=False,
                            )
                        )
                        raise RimeError(f"Rime websocket error: {message['error']}")
                    if "done" in message or "cancelled" in message:
                        break
                if buffer:
                    chunk_queue.put_nowait(buffer)
                logger.info(
                    "Rime response: "
                    + json.dumps(
                        {
                            "transport": "websocket",
                            "context_id": context_id,
                            "request_id": request_id,
                            "total_bytes": total_bytes,
                            "frames": frames,
                            "ttfb_ms": (
                                round((first_chunk_at - started_at) * 1000)
                                if first_chunk_at
                                else None
                            ),
                            "ttlb_ms": round((time.monotonic() - started_at) * 1000),
                            **ids,
                        },
                        ensure_ascii=False,
                    )
                )
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)

    def get_request_body(self, text):
        speed_alpha = self.speed_alpha if self.speed_alpha else RIME_DEFAULT_SPEED_ALPHA
        reduce_latency = self.reduce_latency if self.reduce_latency else RIME_DEFAULT_REDUCE_LATENCY

        body = {
            "text": strip_non_speech(text),
            "speaker": self.speaker,
            "samplingRate": self.sampling_rate,
            "speedAlpha": speed_alpha,
            "reduceLatency": reduce_latency,
        }

        if self.model_id:
            body["modelId"] = self.model_id

        return body
