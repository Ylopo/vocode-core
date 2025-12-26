import asyncio
import base64
import json
from typing import Optional

import websockets
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
    Transcription,
)
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

ASSEMBLYAI_WS_URL = "wss://api.assemblyai.com/v2/realtime/ws"

class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = "c02db566717b478db6117f7bec4c02d0" or getattr(transcriber_config, 'api_key', None) or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise Exception("Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter")
        self._ended = False

    async def ready(self):
        return True

    async def terminate(self):
        self._ended = True
        await super().terminate()

    def get_assemblyai_url(self):
        params = {
            "sample_rate": self.transcriber_config.sampling_rate
        }
        if getattr(self.transcriber_config, "word_boost", None):
            params["word_boost"] = json.dumps(self.transcriber_config.word_boost)
        return f"{ASSEMBLYAI_WS_URL}?{ '&'.join([f'{k}={v}' for k, v in params.items()]) }"

    async def process(self):
        url = self.get_assemblyai_url()
        logger.info(f"Connecting to AssemblyAI at {url}")
        silence_ms = getattr(self.transcriber_config, "end_utterance_silence_threshold_milliseconds", None)
        silence_msg = (
            json.dumps({"end_utterance_silence_threshold": silence_ms})
            if silence_ms is not None else None
        )

        async def _run_loop(self):
            await self.process()

        async with websockets.connect(
            url,
            extra_headers={"Authorization": self.api_key},
            ping_interval=5,
            ping_timeout=20,
            max_size=1024*1024,
        ) as ws:
            if silence_msg:
                await ws.send(silence_msg)
            logger.info("Sent silence threshold config to AssemblyAI")

            async def sender():
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self._input_queue.get(), timeout=5)
                    except asyncio.TimeoutError:
                        break
                    if self.transcriber_config.audio_encoding != AudioEncoding.LINEAR16:
                        logger.error("AssemblyAI requires LINEAR16 audio encoding")
                        continue
                    # Base64 encode the audio bytes for AssemblyAI
                    audio_b64 = base64.b64encode(data).decode("utf-8")
                    await ws.send(json.dumps({"audio_data": audio_b64}))
                # Terminate gracefully as per docs
                await ws.send(json.dumps({"terminate_session": True}))
                logger.info("Sent terminate_session to AssemblyAI websocket")

            async def receiver():
                while not self._ended:
                    try:
                        msg = await ws.recv()
                    except (websockets.ConnectionClosed, asyncio.TimeoutError):
                        break
                    data = json.loads(msg)
                    if "error" in data and data["error"]:
                        logger.error(f"AssemblyAI error: {data['error']}")
                        break
                    # Handle PartialTranscript / FinalTranscript events
                    if "message_type" in data:
                        if data["message_type"] in ("PartialTranscript", "FinalTranscript"):
                            text = data.get("text", "")
                            if text:
                                self.produce_nonblocking(
                                    Transcription(
                                        message=text,
                                        confidence=data.get("confidence", 1.0),
                                        is_final=(data["message_type"] == "FinalTranscript"),
                                    )
                                )

            await asyncio.gather(sender(), receiver())