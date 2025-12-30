import asyncio
import audioop
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

ASSEMBLYAI_WS_URL =  "wss://streaming.assemblyai.com/v3/ws"

class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = (
            "571d6c90beeb4ecf97bd6b288dc31764"
            or getattr(transcriber_config, 'api_key', None)
            or getenv("ASSEMBLY_AI_API_KEY")
        )
        if not self.api_key:
            raise Exception("Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter")
        self._ended = False

    async def ready(self):
        return True

    async def terminate(self):
        logger.info("Terminating AssemblyAITranscriber")
        self._ended = True
        await super().terminate()

    def get_assemblyai_url(self):
        params = {
            "sample_rate": self.transcriber_config.sampling_rate, 
            "model": "universal"
        }
        if getattr(self.transcriber_config, "word_boost", None):
            params["word_boost"] = json.dumps(self.transcriber_config.word_boost)
        url = f"{ASSEMBLYAI_WS_URL}?{ '&'.join([f'{k}={v}' for k, v in params.items()]) }"
        logger.debug(f"Generated AssemblyAI URL: {url}")
        return url

    async def _run_loop(self):
        while not self._ended:
            try:
                logger.info("AssemblyAITranscriber run loop started")
                await self.process()
            except Exception as e:
                logger.error(f"AssemblyAI connection error: {e}, reconnecting in 5 seconds...")
                if not self._ended:
                    await asyncio.sleep(5)

    async def process(self):
        url = self.get_assemblyai_url()
        logger.info(f"Connecting to AssemblyAI at {url}")
        silence_ms = getattr(self.transcriber_config, "end_utterance_silence_threshold_milliseconds", None)
        silence_msg = (
            json.dumps({"end_utterance_silence_threshold": silence_ms})
            if silence_ms is not None else None
        )

        async with websockets.connect(
            url,
            extra_headers={"Authorization": self.api_key},
            ping_interval=5,
            ping_timeout=20,
            max_size=1024*1024,
        ) as ws:
            logger.info("Connected to AssemblyAI websocket")
            if silence_msg:
                await ws.send(silence_msg)
                logger.info(f"Sent silence threshold config: {silence_msg}")

            async def sender():
                logger.info("AssemblyAI sender coroutine started")
                MIN_CHUNK_SIZE = 1600  # 50ms at 16kHz, 16-bit mono LINEAR16

                audio_buffer = b""
                chunk_num = 0

                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self._input_queue.get(), timeout=5)
                        logger.debug(
                            f"Got raw input audio: size={len(data)}, encoding={self.transcriber_config.audio_encoding}, sample_rate={self.transcriber_config.sampling_rate}"
                        )

                        # 1. Convert mulaw to LINEAR16, if necessary
                        if self.transcriber_config.audio_encoding != AudioEncoding.LINEAR16:
                            logger.warning("AssemblyAI requires LINEAR16 audio, converting from MULAW")
                            data = audioop.ulaw2lin(data, 2)  # still at original sample rate

                        # 2. Upsample to 16kHz, if necessary
                        if self.transcriber_config.sampling_rate != 16000:
                            logger.warning(
                                f"Upsampling audio from {self.transcriber_config.sampling_rate}Hz to 16000Hz for AssemblyAI."
                            )
                            data, _ = audioop.ratecv(data, 2, 1, self.transcriber_config.sampling_rate, 16000, None)

                        audio_buffer += data
                        while len(audio_buffer) >= MIN_CHUNK_SIZE:
                            chunk_num += 1
                            chunk = audio_buffer[:MIN_CHUNK_SIZE]
                            audio_buffer = audio_buffer[MIN_CHUNK_SIZE:]
                            logger.debug(
                                f"Sending raw audio chunk #{chunk_num} to AssemblyAI (size: {len(chunk)} bytes)"
                            )
                            await ws.send(chunk)

                    except asyncio.TimeoutError:
                        logger.warning("Sender timed out waiting for audio data")
                        break

                # Send any leftover audio at stream end
                if audio_buffer:
                    chunk_num += 1
                    logger.debug(
                        f"Sending final raw audio chunk #{chunk_num} to AssemblyAI (size: {len(audio_buffer)} bytes)"
                    )
                    await ws.send(audio_buffer)

                logger.info("Sender done sending audio, sending terminate_session")
                # Terminate gracefully as per docs
                await ws.send(json.dumps({"terminate_session": True}))
                logger.info("Sent terminate_session to AssemblyAI websocket")
                logger.info("Sender coroutine exiting")

            async def receiver():
                logger.info("AssemblyAI receiver coroutine started")
                msg_num = 0
                while not self._ended:
                    try:
                        msg = await ws.recv()
                        msg_num += 1
                        logger.debug(f"Received websocket message #{msg_num}")
                    except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                        logger.warning(f"Receiver websocket closed or timed out: {e}")
                        break
                    try:
                        data = json.loads(msg)
                        logger.debug(f"Full AssemblyAI message: {msg[:400]}...")
                    except Exception as e:
                        logger.error(f"Failed to parse AssemblyAI message: {e}. Raw: '{msg[:400]}'")
                        continue

                    if "error" in data and data["error"]:
                        logger.error(f"AssemblyAI error: {data['error']}")
                        break
                    # Handle PartialTranscript / FinalTranscript events
                    if "message_type" in data:
                        logger.info(f"AssemblyAI {data['message_type']} received")
                        if data["message_type"] in ("PartialTranscript", "FinalTranscript"):
                            text = data.get("text", "")
                            confidence = data.get("confidence", 1.0)
                            logger.info(f"Transcription ({data['message_type']}): '{text}' (confidence={confidence})")
                            if text:
                                self.produce_nonblocking(
                                    Transcription(
                                        message=text,
                                        confidence=confidence,
                                        is_final=(data["message_type"] == "FinalTranscript"),
                                    )
                                )

            logger.info("Starting AssemblyAI sender and receiver tasks")
            await asyncio.gather(sender(), receiver())
            logger.info("AssemblyAI sender and receiver tasks completed")