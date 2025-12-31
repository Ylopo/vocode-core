import asyncio
import audioop
import json
from typing import Optional
from urllib.parse import urlencode
import queue

import numpy as np
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
from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

ASSEMBLY_AI_URL = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.buffer = bytearray()
        self.audio_cursor = 0
        
        # Audio queue for sender
        self.audio_queue = asyncio.Queue()

        if isinstance(
            self.transcriber_config.endpointing_config,
            (TimeEndpointingConfig, PunctuationEndpointingConfig),
        ):
            self.transcriber_config.end_utterance_silence_threshold_milliseconds = int(
                self.transcriber_config.endpointing_config.time_cutoff_seconds * 1000
            )
        
        # Updated message format for v3
        self.terminate_msg = json.dumps({"type": "Terminate"})

    async def ready(self):
        return True

    async def _run_loop(self):
        await self.process()

    def send_audio(self, chunk):
        if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            sample_width = 1
            if isinstance(chunk, np.ndarray):
                chunk = chunk.astype(np.int16)
                chunk = chunk.tobytes()
            chunk = audioop.ulaw2lin(chunk, sample_width)

        # Ensure chunk is bytes
        if isinstance(chunk, np.ndarray):
            chunk = chunk.astype(np.int16).tobytes()

        self.buffer.extend(chunk)

        if (
            len(self.buffer) / (2 * self.transcriber_config.sampling_rate)
        ) >= self.transcriber_config.buffer_size_seconds:
            # Put audio in queue for sender
            try:
                self.audio_queue.put_nowait(bytes(self.buffer))
            except asyncio.QueueFull:
                logger.warning("Audio queue is full, dropping audio chunk")
            self.buffer = bytearray()

    async def terminate(self):
        self._ended = True
        await super().terminate()

    def get_assembly_ai_url(self):
        url_params = {"sample_rate": self.transcriber_config.sampling_rate}
        # Note: Check AssemblyAI v3 docs for supported parameters
        # word_boost may not be available in v3
        return ASSEMBLY_AI_URL + f"?{urlencode(url_params)}"

    async def sender(self, ws):
        """Sends audio data to AssemblyAI WebSocket"""
        logger.info("AssemblyAI sender coroutine started")
        loop_num = 0
        
        while not self._ended:
            loop_num += 1
            logger.debug(f"Sender coroutine main loop iteration {loop_num} started")
            
            try:
                # Wait for audio data with timeout
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=0.1
                )
                
                if audio_data:
                    logger.debug(f"Sender got audio: {len(audio_data)} bytes from queue")
                    duration = len(audio_data) / (2 * self.transcriber_config.sampling_rate)
                    logger.info(f"Sender sending {len(audio_data)} bytes ({duration:.3f} sec) to AssemblyAI")
                    
                    # Send as binary WebSocket frame
                    await ws.send(audio_data)
                    
                    logger.info(f"Sender sent {len(audio_data)} bytes successfully")
                    
            except asyncio.TimeoutError:
                # No audio data available, continue loop
                continue
            except Exception as e:
                logger.error(f"Sender failed to send chunk: {e}")
                break
        
        # Send termination message
        try:
            logger.info("Sending termination message")
            await ws.send(self.terminate_msg)
            logger.info("Termination message sent")
        except Exception as e:
            logger.error(f"Sender failed to send terminate_session: {e}")
        
        logger.debug("Terminating AssemblyAI transcriber sender")

    async def receiver(self, ws):
        """Receives and processes messages from AssemblyAI WebSocket"""
        logger.info("AssemblyAI receiver coroutine started")
        
        try:
            async for message in ws:
                if self._ended:
                    break
                    
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    logger.debug(f"Received message type: {msg_type}")
                    
                    if msg_type == "Begin":
                        session_id = data.get('id')
                        logger.info(f"Session began: {session_id}")
                        
                    elif msg_type == "Turn":
                        transcript = data.get('transcript', '')
                        utterance = data.get('utterance', '')
                        is_final = data.get('end_of_turn', False)
                        confidence = data.get('end_of_turn_confidence', 0)
                        
                        logger.debug(f"Turn - transcript: '{transcript}', final: {is_final}, confidence: {confidence}")
                        
                        # Use utterance for pre-emptive generation if available
                        text_to_process = utterance if utterance else transcript
                        
                        if text_to_process:
                            transcription = Transcription(
                                message=text_to_process,
                                confidence=confidence,
                                is_final=is_final,
                            )
                            self.produce_nonblocking(transcription)
                        
                    elif msg_type == "Termination":
                        reason = data.get('message', 'Unknown')
                        logger.info(f"Session terminated: {reason}")
                        break
                        
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Receiver error: {e}")
        
        logger.debug("Terminating AssemblyAI transcriber receiver")

    async def process(self):
        """Main processing method that manages WebSocket connection and tasks"""
        self.audio_cursor = 0
        URL = self.get_assembly_ai_url()
        logger.info(f"Connecting to AssemblyAI at {URL}")
        
        try:
            async with websockets.connect(
                URL,
                extra_headers=(("Authorization", self.api_key),),
                ping_interval=5,
                ping_timeout=20,
            ) as ws:
                logger.info("Connected to AssemblyAI")
                await asyncio.sleep(0.1)  # Small delay for connection stability
                
                # Create sender and receiver tasks
                sender_task = asyncio.create_task(self.sender(ws))
                receiver_task = asyncio.create_task(self.receiver(ws))
                
                try:
                    # Run both tasks concurrently
                    await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error in process tasks: {e}")
                finally:
                    # Cancel any remaining tasks
                    if not sender_task.done():
                        sender_task.cancel()
                    if not receiver_task.done():
                        receiver_task.cancel()
                    
                    # Wait for tasks to complete cancellation
                    await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
                    
        except websockets.exceptions.InvalidURI as e:
            logger.error(f"Invalid WebSocket URI: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in process: {e}")
        finally:
            logger.info("AssemblyAI transcriber process ended")
