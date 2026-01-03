"""
Phone Agent using Local STT/TTS

Uses:
- SIM7600 modem for calls
- Local Whisper for speech recognition
- Claude Haiku for conversation
- Local Piper for text-to-speech
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
import os

from sim7600_modem import SIM7600Modem, CallState
from audio_router_sim7600 import AudioRouterSIM7600
from conversation_local import LocalConversationEngine, ConversationConfig, ConversationState

import config

logger = logging.getLogger(__name__)


@dataclass
class CallRequest:
    phone: str
    objective: str
    context: dict = field(default_factory=dict)


@dataclass
class CallResult:
    success: bool
    summary: str
    transcript: list = field(default_factory=list)
    duration_seconds: float = 0.0


class PhoneAgentLocal:
    """Phone agent using local STT/TTS pipeline"""

    def __init__(self):
        self.modem = SIM7600Modem()
        self.audio = AudioRouterSIM7600()
        self.conversation = LocalConversationEngine()

        self._call_active = False
        self._audio_output_queue = asyncio.Queue()

        # External callbacks (for web UI etc)
        self._external_state_callback = None
        self._external_transcript_callback = None

    def on_state_change(self, callback):
        """Register external callback for state changes"""
        self._external_state_callback = callback

    def on_transcript(self, callback):
        """Register external callback for transcript updates"""
        self._external_transcript_callback = callback

    async def call(self, request: CallRequest) -> CallResult:
        """
        Make a phone call with local STT/TTS.

        Args:
            request: Call request with phone, objective, context

        Returns:
            CallResult with success, summary, transcript
        """
        logger.info(f"Starting call to {request.phone}")
        logger.info(f"Objective: {request.objective}")

        start_time = time.time()
        recording_path = None

        try:
            # 1. Connect modem
            if not self.modem.connect():
                raise Exception("Failed to connect to SIM7600 modem")

            # 2. Initialize conversation engine
            self.conversation.initialize()

            # Set up callbacks
            self.conversation.on_state_change(self._on_state_change)
            self.conversation.on_transcript(self._on_transcript)

            # 3. Start audio
            if not self.audio.start():
                raise Exception("Failed to start audio routing")

            self.audio.start_recording()

            # 4. Dial
            if not self.modem.dial(request.phone):
                raise Exception("Failed to dial")

            # Wait for call to connect
            call_state = await self._wait_for_connect(timeout=60)

            if call_state != CallState.CONNECTED:
                raise Exception("Call did not connect")

            logger.info("Call connected!")
            self._call_active = True

            # Wait for audio to stabilize after connection
            await asyncio.sleep(0.5)

            # 5. Start conversation (sets LLM objective)
            conv_config = ConversationConfig(
                objective=request.objective,
                context=request.context
            )
            self.conversation.start(conv_config)

            # 6. AI speaks first - generate and play greeting
            logger.info("Generating initial greeting...")
            greeting_text = self.conversation.get_initial_greeting()
            greeting_audio = self.conversation.synthesize_greeting(greeting_text)

            if greeting_audio:
                logger.info(f"Playing greeting ({len(greeting_audio)} bytes)")
                self.conversation.set_speaking(True)
                await self.audio.write_audio_async(greeting_audio)
                await asyncio.sleep(0.3)  # Small buffer after audio
                self.conversation.set_speaking(False)

            # Clear any buffered audio from before/during greeting
            # This prevents the user's initial "hello" from being processed
            self.audio.clear_input_buffer()

            # 7. Main loop - process audio (now listening for response)
            await self._run_conversation_loop()

            # 7. Get result
            result = self.conversation.get_result()

            return CallResult(
                success=result.success,
                summary=result.summary,
                transcript=result.transcript,
                duration_seconds=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Call failed: {e}")
            return CallResult(
                success=False,
                summary=str(e),
                transcript=[],
                duration_seconds=time.time() - start_time
            )

        finally:
            # Cleanup
            self._call_active = False
            self.conversation.stop()

            # Normal cleanup - hang up and disconnect
            try:
                self.modem.hangup()
            except:
                pass

            try:
                self.modem.disconnect()
            except:
                pass

            # Save recording
            recording_path = self.audio.stop_recording()
            self.audio.stop()

            # Save call log
            self._save_call_log(request, self.conversation.get_result(), recording_path)

    async def _wait_for_connect(self, timeout: float = 60) -> CallState:
        """Wait for call to connect"""
        start = time.time()

        while time.time() - start < timeout:
            call_info = self.modem.get_call_info()

            if call_info is None:
                return CallState.IDLE

            if call_info.state == CallState.CONNECTED:
                return CallState.CONNECTED

            if call_info.state == CallState.ENDED:
                return CallState.ENDED

            await asyncio.sleep(0.5)

        return CallState.IDLE

    async def _run_conversation_loop(self):
        """Main conversation loop"""
        logger.info("Starting conversation loop")

        # Create tasks
        audio_task = asyncio.create_task(self._audio_input_loop())
        output_task = asyncio.create_task(self._audio_output_loop())
        monitor_task = asyncio.create_task(self._monitor_call())

        try:
            # Run until call ends
            await asyncio.gather(audio_task, output_task, monitor_task)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel all tasks
            audio_task.cancel()
            output_task.cancel()
            monitor_task.cancel()

    async def _audio_input_loop(self):
        """Read audio from phone and process"""
        logger.info("Audio input loop started")

        while self._call_active:
            # Read audio chunk
            audio_bytes = await self.audio.read_audio()

            if audio_bytes:
                # Convert to numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)

                # Process through conversation engine
                response_audio = self.conversation.process_audio(audio_chunk)

                if response_audio:
                    # Queue for output
                    await self._audio_output_queue.put(response_audio)

            await asyncio.sleep(0.01)

    async def _audio_output_loop(self):
        """Play audio responses to phone"""
        logger.info("Audio output loop started")

        while self._call_active:
            try:
                # Get audio to play (with timeout)
                response_audio = await asyncio.wait_for(
                    self._audio_output_queue.get(),
                    timeout=0.1
                )

                if response_audio:
                    logger.info(f"Playing response audio ({len(response_audio)} bytes)")

                    # Mark as speaking (echo suppression)
                    self.conversation.set_speaking(True)

                    # Play audio (blocking call - waits until audio is played)
                    await self.audio.write_audio_async(response_audio)

                    # Small buffer after audio finishes
                    await asyncio.sleep(0.3)

                    # Done speaking
                    self.conversation.set_speaking(False)

                    # Clear any buffered audio to prevent echo/stale data
                    self.audio.clear_input_buffer()

                    # Check if conversation completed
                    if self.conversation.state == ConversationState.COMPLETED:
                        logger.info("Conversation completed")
                        await asyncio.sleep(1)  # Let final audio play
                        self._call_active = False

                    # Check if callback was requested
                    elif self.conversation.state == ConversationState.TRANSFERRING:
                        callback_number = self.conversation._transfer_number
                        logger.info(f"Callback requested - caller wants to speak with someone")
                        await asyncio.sleep(2)  # Let the callback message play fully
                        # Log the callback request (the call ends normally)
                        logger.info(f"Callback logged - you should call them back")
                        self._call_active = False

            except asyncio.TimeoutError:
                continue

    async def _monitor_call(self):
        """Monitor call state"""
        while self._call_active:
            call_info = self.modem.get_call_info()

            if call_info is None or call_info.state == CallState.ENDED:
                logger.info("Call ended")
                self._call_active = False
                break

            await asyncio.sleep(0.5)

    def _on_state_change(self, state: ConversationState):
        """Handle conversation state changes"""
        logger.info(f"Conversation state: {state.value}")
        # Call external callback if registered
        if self._external_state_callback:
            self._external_state_callback(state)

    def _on_transcript(self, role: str, text: str):
        """Handle transcript updates"""
        logger.info(f"{role.title()}: {text}")
        # Call external callback if registered
        if self._external_transcript_callback:
            self._external_transcript_callback(role, text)

    def _save_call_log(self, request: CallRequest, result, recording_path: Optional[str]):
        """Save call log to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "phone": request.phone,
            "objective": request.objective,
            "context": request.context,
            "success": result.success,
            "summary": result.summary,
            "transcript": result.transcript,
            "recording_path": recording_path,
            "duration_seconds": result.duration_seconds,
            "engine": "local"
        }

        log_path = os.path.join(
            config.CALLS_DIR,
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        os.makedirs(config.CALLS_DIR, exist_ok=True)

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Call log saved to {log_path}")


# Main entry point
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    agent = PhoneAgentLocal()

    request = CallRequest(
        phone="2283430054",
        objective="You are calling from Dr. Smith's dental office to confirm an appointment scheduled for tomorrow at 2:30 PM.",
        context={"office": "Dr. Smith's Dental Office", "appointment_time": "2:30 PM tomorrow"}
    )

    result = await agent.call(request)

    print(f"\n{'='*60}")
    print(f"Call Result:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Summary: {result.summary}")
    print(f"\nTranscript:")
    for msg in result.transcript:
        print(f"  {msg['role']}: {msg['content']}")


if __name__ == "__main__":
    asyncio.run(main())
