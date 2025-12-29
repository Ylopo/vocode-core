import asyncio
import audioop
import base64
import json
import time
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

ASSEMBLYAI_WS_URL = "wss://api.assemblyai.com/v2/stream/ws"
NUM_RESTARTS = 5


class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = "c02db566717b478db6117f7bec4c02d0" or getattr(transcriber_config, 'api_key', None) or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key must be provided via parameter, "
                "transcriber_config, or ASSEMBLY_AI_API_KEY environment variable"
            )
        self._ended = False
        self.is_ready = False
        
        # Debug statistics
        self.connection_start_time = None
        self.total_audio_bytes_sent = 0
        self.total_messages_received = 0
        self.partial_transcripts = 0
        self.final_transcripts = 0
        self.last_activity_time = None
        self.session_id = None

    async def ready(self):
        return self.is_ready

    async def terminate(self):
        logger.debug("AssemblyAITranscriber.terminate() called")
        self._ended = True
        await super().terminate()
        self._log_session_summary()

    def get_assemblyai_url(self):
        params = {
            "sample_rate": self.transcriber_config.sampling_rate
        }
        if getattr(self.transcriber_config, "word_boost", None):
            params["word_boost"] = json.dumps(self.transcriber_config.word_boost)
        
        url = f"{ASSEMBLYAI_WS_URL}?{ '&'.join([f'{k}={v}' for k, v in params.items()]) }"
        logger.debug(f"Generated AssemblyAI URL with params: {params}")
        return url

    async def _run_loop(self):
        """Main loop with restart logic"""
        logger.info("AssemblyAITranscriber._run_loop() started")
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            try:
                logger.debug(f"Starting AssemblyAI transcription process (attempt {restarts + 1}/{NUM_RESTARTS})")
                await self.process()
                
                # If process returns without error, connection was closed normally
                logger.info("AssemblyAI process completed normally")
                break
                
            except websockets.ConnectionClosedError as e:
                logger.error(f"AssemblyAI WebSocket connection closed: {e.code} - {e.reason}")
                if not self._ended:
                    restarts += 1
                    logger.warning(f"Restarting AssemblyAI, num_restarts: {restarts}")
                    await asyncio.sleep(2 ** restarts)  # Exponential backoff
                continue
                
            except Exception as e:
                logger.error(f"AssemblyAI connection error: {type(e).__name__}: {e}")
                if not self._ended:
                    restarts += 1
                    logger.warning(f"Restarting AssemblyAI, num_restarts: {restarts}")
                    await asyncio.sleep(2 ** restarts)  # Exponential backoff
                continue

        if not self._ended and restarts >= NUM_RESTARTS:
            logger.error("AssemblyAI connection failed after maximum restarts")
            raise ConnectionError("Failed to establish AssemblyAI connection after multiple attempts")

    async def process(self):
        """Process audio stream through AssemblyAI WebSocket"""
        self.is_ready = False
        self.connection_start_time = time.time()
        self.last_activity_time = time.time()
        url = self.get_assemblyai_url()
        
        logger.info(f"Connecting to AssemblyAI at {url}")
        logger.debug(f"Audio encoding: {self.transcriber_config.audio_encoding}, "
                    f"Sample rate: {self.transcriber_config.sampling_rate}")
        
        silence_ms = getattr(self.transcriber_config, "end_utterance_silence_threshold_milliseconds", None)
        if silence_ms:
            logger.debug(f"Setting silence threshold: {silence_ms}ms")
        
        silence_msg = (
            json.dumps({"end_utterance_silence_threshold": silence_ms})
            if silence_ms is not None else None
        )

        try:
            connection_start = time.time()
            async with websockets.connect(
                url,
                extra_headers={"Authorization": self.api_key},
                ping_interval=5,
                ping_timeout=20,
                max_size=1024*1024,
                close_timeout=10,
            ) as ws:
                connection_time = time.time() - connection_start
                logger.info(f"Connected to AssemblyAI websocket in {connection_time:.2f}s")
                self.is_ready = True
                self.last_activity_time = time.time()
                
                # Send initial configuration
                if silence_msg:
                    logger.debug("Sending silence threshold configuration")
                    await ws.send(silence_msg)
                    logger.debug("Silence threshold configuration sent")

                async def sender():
                    """Send audio data to AssemblyAI"""
                    logger.debug("AssemblyAI sender task started")
                    audio_chunks_sent = 0
                    last_log_time = time.time()
                    
                    while not self._ended:
                        try:
                            # Get audio data from input queue
                            get_start = time.time()
                            data = await asyncio.wait_for(self._input_queue.get(), timeout=5)
                            queue_wait_time = time.time() - get_start
                            
                            if queue_wait_time > 1.0:
                                logger.warning(f"Input queue wait time high: {queue_wait_time:.2f}s")
                            
                            audio_chunks_sent += 1
                            self.total_audio_bytes_sent += len(data)
                            self.last_activity_time = time.time()
                            
                            # Convert audio if needed
                            original_encoding = self.transcriber_config.audio_encoding
                            if original_encoding != AudioEncoding.LINEAR16:
                                if original_encoding == AudioEncoding.MULAW:
                                    logger.debug(f"Converting μ-law to LINEAR16 (chunk {audio_chunks_sent})")
                                    data = audioop.ulaw2lin(data, 2)
                                    data = audioop.ratecv(data, 2, 1, 8000, 16000, None)[0]
                                else:
                                    logger.error(f"Unsupported audio encoding: {original_encoding}")
                                    continue
                            
                            # Encode and send
                            encode_start = time.time()
                            audio_b64 = base64.b64encode(data).decode("utf-8")
                            encode_time = time.time() - encode_start
                            
                            message = json.dumps({"audio_data": audio_b64})
                            send_start = time.time()
                            await ws.send(message)
                            send_time = time.time() - send_start
                            
                            # Log progress periodically
                            current_time = time.time()
                            if current_time - last_log_time > 10.0:  # Every 10 seconds
                                logger.info(
                                    f"Sender stats: {audio_chunks_sent} chunks, "
                                    f"{self.total_audio_bytes_sent / 1024:.1f}KB sent, "
                                    f"encode: {encode_time*1000:.1f}ms, "
                                    f"send: {send_time*1000:.1f}ms"
                                )
                                last_log_time = current_time
                                
                        except asyncio.TimeoutError:
                            # Normal timeout when no data is available
                            if not self._ended:
                                # Log if we haven't sent data for a while
                                idle_time = time.time() - self.last_activity_time
                                if idle_time > 30.0:
                                    logger.warning(f"No audio data for {idle_time:.1f}s")
                            break
                        except Exception as e:
                            logger.error(f"Error in sender task: {type(e).__name__}: {e}")
                            break
                    
                    logger.debug("Sender task ending")
                    # Terminate gracefully as per docs
                    if not self._ended:
                        logger.debug("Sending terminate_session message")
                        await ws.send(json.dumps({"terminate_session": True}))
                        logger.info("Sent terminate_session to AssemblyAI")
                    logger.debug(f"Sender task completed. Total chunks: {audio_chunks_sent}")

                async def receiver():
                    """Receive transcriptions from AssemblyAI"""
                    logger.debug("AssemblyAI receiver task started")
                    last_heartbeat_time = time.time()
                    
                    while not self._ended:
                        try:
                            receive_start = time.time()
                            msg = await ws.recv()
                            receive_time = time.time() - receive_start
                            
                            self.total_messages_received += 1
                            self.last_activity_time = time.time()
                            
                            # Log periodic heartbeats
                            current_time = time.time()
                            if current_time - last_heartbeat_time > 30.0:
                                logger.debug(f"Receiver active, messages: {self.total_messages_received}")
                                last_heartbeat_time = current_time
                            
                            if receive_time > 1.0:
                                logger.warning(f"WebSocket receive time high: {receive_time:.2f}s")
                            
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse AssemblyAI response: {e}")
                                logger.debug(f"Raw message: {msg[:200]}...")
                                continue
                            
                            # Log message type for debugging
                            message_type = data.get("message_type", "Unknown")
                            logger.debug(f"Received message type: {message_type}, "
                                       f"size: {len(msg)} bytes, "
                                       f"receive_time: {receive_time*1000:.1f}ms")
                            
                            if "error" in data and data["error"]:
                                logger.error(f"AssemblyAI error: {data['error']}")
                                # Don't break immediately, let sender handle termination
                                continue
                            
                            if message_type == "SessionBegins":
                                self.session_id = data.get("session_id", "unknown")
                                logger.info(f"AssemblyAI session started. Session ID: {self.session_id}")
                                continue
                                
                            elif message_type == "PartialTranscript":
                                self.partial_transcripts += 1
                                text = data.get("text", "").strip()
                                confidence = data.get("confidence", 0.0)
                                
                                if text:
                                    logger.debug(f"Partial transcript: '{text}' (confidence: {confidence:.2f})")
                                    self.produce_nonblocking(
                                        Transcription(
                                            message=text,
                                            confidence=confidence,
                                            is_final=False,
                                        )
                                    )
                                    
                            elif message_type == "FinalTranscript":
                                self.final_transcripts += 1
                                text = data.get("text", "").strip()
                                confidence = data.get("confidence", 0.0)
                                
                                if text:
                                    logger.info(f"Final transcript: '{text}' (confidence: {confidence:.2f})")
                                    self.produce_nonblocking(
                                        Transcription(
                                            message=text,
                                            confidence=confidence,
                                            is_final=True,
                                        )
                                    )
                                    
                            elif message_type == "SessionTerminated":
                                logger.info("AssemblyAI session terminated by server")
                                break
                                
                            else:
                                logger.debug(f"Ignoring message type: {message_type}")
                                
                        except websockets.ConnectionClosed as e:
                            logger.info(f"AssemblyAI WebSocket connection closed in receiver: {e.code} - {e.reason}")
                            break
                        except asyncio.TimeoutError:
                            logger.warning("WebSocket receive timeout")
                            # Check if we should continue
                            idle_time = time.time() - self.last_activity_time
                            if idle_time > 60.0:
                                logger.error(f"No activity for {idle_time:.1f}s, closing connection")
                                break
                            continue
                        except Exception as e:
                            logger.error(f"Error in receiver task: {type(e).__name__}: {e}")
                            break
                    
                    logger.debug("Receiver task ending")
                    logger.info(f"Receiver completed. Messages: {self.total_messages_received}")

                # Run both tasks concurrently
                logger.debug("Starting sender and receiver tasks")
                sender_task = asyncio.create_task(sender())
                receiver_task = asyncio.create_task(receiver())
                
                try:
                    # Wait for both tasks to complete
                    await asyncio.gather(sender_task, receiver_task)
                except asyncio.CancelledError:
                    logger.debug("Tasks cancelled")
                    # Cancel both tasks if needed
                    sender_task.cancel()
                    receiver_task.cancel()
                    await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
                
                logger.debug("Both sender and receiver tasks completed")
                
        except websockets.ConnectionClosedError as e:
            logger.error(f"Failed to connect to AssemblyAI: {e.code} - {e.reason}")
            raise
        except asyncio.TimeoutError:
            logger.error("Timeout connecting to AssemblyAI")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AssemblyAI process: {type(e).__name__}: {e}")
            raise
        finally:
            self.is_ready = False
            self._log_connection_summary()

    def _log_connection_summary(self):
        """Log connection statistics"""
        if self.connection_start_time:
            duration = time.time() - self.connection_start_time
            logger.info(
                f"AssemblyAI connection summary - "
                f"Duration: {duration:.1f}s, "
                f"Audio sent: {self.total_audio_bytes_sent / 1024:.1f}KB, "
                f"Messages received: {self.total_messages_received}, "
                f"Partial transcripts: {self.partial_transcripts}, "
                f"Final transcripts: {self.final_transcripts}"
            )

    def _log_session_summary(self):
        """Log session summary when terminating"""
        logger.info(
            "AssemblyAI transcription session ended - "
            f"Total audio sent: {self.total_audio_bytes_sent / 1024:.1f}KB, "
            f"Total messages: {self.total_messages_received}, "
            f"Partial: {self.partial_transcripts}, "
            f"Final: {self.final_transcripts}"
        )

    # Additional debug methods
    def get_debug_info(self):
        """Get current debug information"""
        info = {
            "is_ready": self.is_ready,
            "ended": self._ended,
            "total_audio_bytes_sent": self.total_audio_bytes_sent,
            "total_messages_received": self.total_messages_received,
            "partial_transcripts": self.partial_transcripts,
            "final_transcripts": self.final_transcripts,
            "session_id": self.session_id,
        }
        
        if self.connection_start_time:
            info["connection_duration"] = time.time() - self.connection_start_time
        if self.last_activity_time:
            info["seconds_since_last_activity"] = time.time() - self.last_activity_time
            
        return info

    def log_status(self):
        """Log current status for debugging"""
        debug_info = self.get_debug_info()
        logger.debug(f"AssemblyAI Transcriber Status: {json.dumps(debug_info, indent=2)}")