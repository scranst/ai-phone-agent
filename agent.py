"""
AI Phone Agent - Main orchestrator

Brings together:
- Call Controller (FaceTime/Continuity)
- Audio Router (BlackHole)
- Conversation Engine (OpenAI Realtime API)

To make AI-powered phone calls.
"""

import asyncio
import logging
import subprocess
import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from call_controller import CallController, CallState
from audio_router import AudioRouter
from conversation import ConversationEngine, ConversationConfig, ConversationStatus, ConversationResult
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    collected_info: dict
    transcript: list
    recording_path: Optional[str]
    duration_seconds: float
    phone: str
    objective: str


class PhoneAgent:
    def __init__(self):
        self.call_controller = CallController()
        self.audio_router = AudioRouter()
        self.conversation: Optional[ConversationEngine] = None
        self._current_request: Optional[CallRequest] = None
        self._is_running = False
        self._original_output_device: Optional[str] = None
        self._original_input_device: Optional[str] = None

        # Event callbacks for UI
        self._on_state_change: Optional[callable] = None
        self._on_transcript: Optional[callable] = None

    def _get_current_audio_device(self, device_type: str) -> Optional[str]:
        """Get current audio device (input or output)"""
        try:
            result = subprocess.run(
                ["SwitchAudioSource", "-c", "-t", device_type],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get current {device_type} device: {e}")
        return None

    def _set_audio_device(self, device_name: str, device_type: str) -> bool:
        """Set audio device (input or output)"""
        try:
            result = subprocess.run(
                ["SwitchAudioSource", "-s", device_name, "-t", device_type],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Switched {device_type} device to: {device_name}")
                return True
            else:
                logger.warning(f"Failed to switch {device_type} to {device_name}: {result.stderr}")
        except FileNotFoundError:
            logger.warning("SwitchAudioSource not installed. Run: brew install switchaudio-osx")
        except Exception as e:
            logger.warning(f"Could not set {device_type} device: {e}")
        return False

    def _get_available_output_devices(self) -> list[str]:
        """Get list of available output devices"""
        try:
            result = subprocess.run(
                ["SwitchAudioSource", "-a", "-t", "output"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
        except Exception:
            pass
        return []

    def _setup_audio_for_call(self):
        """Setup audio for call - using iPhone mic directly, no system audio changes needed"""
        # Save current devices for restoration
        self._original_output_device = self._get_current_audio_device("output")
        self._original_input_device = self._get_current_audio_device("input")

        # No need to change system audio - we capture directly from iPhone mic
        # and output to Mac speakers (which iPhone mic picks up)
        logger.info("Using iPhone mic for input, Mac speakers for output (speakerphone mode)")

        # Kill FaceTime to ensure clean state
        try:
            subprocess.run(["killall", "FaceTime"], capture_output=True, timeout=5)
            subprocess.run(["killall", "FaceTimeNotificationService"], capture_output=True, timeout=5)
            logger.info("Force-killed FaceTime processes")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Could not kill FaceTime: {e}")

        logger.info("Audio configured for call")

    def _restore_audio_after_call(self):
        """Restore original audio devices after call"""
        if self._original_output_device:
            self._set_audio_device(self._original_output_device, "output")
        if self._original_input_device:
            self._set_audio_device(self._original_input_device, "input")

    def on_state_change(self, callback):
        """Register callback for state changes"""
        self._on_state_change = callback

    def on_transcript(self, callback):
        """Register callback for transcript updates"""
        self._on_transcript = callback

    async def call(self, request: CallRequest) -> CallResult:
        """
        Make an AI-powered phone call

        Args:
            request: CallRequest with phone, objective, and context

        Returns:
            CallResult with success status, summary, and collected info
        """
        self._current_request = request
        self._is_running = True
        recording_path = None
        start_time = datetime.now()

        logger.info(f"Starting call to {request.phone}")
        logger.info(f"Objective: {request.objective}")

        # Set up system audio routing for the call
        self._setup_audio_for_call()

        try:
            # Initialize conversation engine
            self.conversation = ConversationEngine()

            # Set up callbacks
            self.conversation.on_audio_output(self._handle_ai_audio)
            self.conversation.on_transcript(self._handle_transcript)
            self.conversation.on_status_change(self._handle_conversation_status)

            # Set up call state callbacks
            self.call_controller.on_state_change(self._handle_call_state)

            # Start audio routing
            if not self.audio_router.start():
                logger.error("Failed to start audio routing")
                # Continue anyway for testing - may work with default devices
                logger.warning("Continuing without BlackHole - audio may not route correctly")

            # Start recording
            self.audio_router.start_recording()

            # Connect to OpenAI
            conv_config = ConversationConfig(
                objective=request.objective,
                context=request.context,
                voice=config.VOICE
            )

            if not await self.conversation.connect(conv_config):
                raise Exception("Failed to connect to OpenAI Realtime API")

            # Start the phone call
            if not await self.call_controller.start_call(request.phone):
                raise Exception("Failed to start phone call")

            # Wait for call to connect
            await self._wait_for_call_connect()

            # Start processing in parallel
            audio_task = asyncio.create_task(self._process_audio())
            message_task = asyncio.create_task(self.conversation.process_messages())

            # Start the conversation
            await self.conversation.start_conversation()

            # Wait for conversation to complete or call to end
            await self._wait_for_completion()

            # Cancel tasks
            audio_task.cancel()
            message_task.cancel()

            try:
                await audio_task
            except asyncio.CancelledError:
                pass
            try:
                await message_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Call failed: {e}")
            return CallResult(
                success=False,
                summary=f"Call failed: {str(e)}",
                collected_info={},
                transcript=[],
                recording_path=None,
                duration_seconds=0,
                phone=request.phone,
                objective=request.objective
            )

        finally:
            # Cleanup
            self._is_running = False

            # End call if still active
            await self.call_controller.end_call()

            # Stop recording and save
            recording_path = self.audio_router.stop_recording()

            # Disconnect from OpenAI
            if self.conversation:
                await self.conversation.disconnect()

            # Stop audio routing
            self.audio_router.stop()

            # Restore original audio devices
            self._restore_audio_after_call()

        # Get results
        duration = (datetime.now() - start_time).total_seconds()
        result = self.conversation.get_result() if self.conversation else ConversationResult(
            success=False, summary="", collected_info={}, transcript=[]
        )

        # Save call log
        self._save_call_log(request, result, recording_path, duration)

        return CallResult(
            success=result.success,
            summary=result.summary,
            collected_info=result.collected_info,
            transcript=result.transcript,
            recording_path=recording_path,
            duration_seconds=duration,
            phone=request.phone,
            objective=request.objective
        )

    async def _wait_for_call_connect(self, timeout: float = 60):
        """Wait for the call to be answered"""
        start = asyncio.get_event_loop().time()

        while self._is_running:
            call_info = self.call_controller.get_call_info()

            if call_info and call_info.state == CallState.CONNECTED:
                logger.info("Call connected!")
                return

            if call_info and call_info.state in [CallState.FAILED, CallState.ENDED]:
                raise Exception("Call failed or ended before connecting")

            if asyncio.get_event_loop().time() - start > timeout:
                raise Exception("Call connection timeout")

            await asyncio.sleep(0.5)

    async def _wait_for_completion(self):
        """Wait for the conversation to complete or call to end"""
        while self._is_running:
            call_info = self.call_controller.get_call_info()
            conv_status = self.conversation.status if self.conversation else None

            # Check if call ended
            if call_info and call_info.state == CallState.ENDED:
                logger.info("Call ended")
                break

            # Check if objective completed
            if conv_status == ConversationStatus.COMPLETED:
                logger.info("Objective completed")
                # Give a moment for goodbye
                await asyncio.sleep(3)
                break

            # Check timeout
            if call_info and call_info.connect_time:
                duration = asyncio.get_event_loop().time() - call_info.connect_time
                if duration > config.MAX_CALL_DURATION:
                    logger.warning("Call duration limit reached")
                    break

            await asyncio.sleep(0.5)

    async def _process_audio(self):
        """Process audio between call and OpenAI"""
        while self._is_running:
            # Read audio from call
            audio = await self.audio_router.read_audio()

            if audio and self.conversation:
                # Send to OpenAI
                await self.conversation.send_audio(audio)

            await asyncio.sleep(0.01)

    def _handle_ai_audio(self, audio_bytes: bytes):
        """Handle audio output from AI"""
        self.audio_router.write_audio(audio_bytes)

    def _handle_transcript(self, role: str, text: str):
        """Handle transcript updates"""
        if self._on_transcript:
            self._on_transcript(role, text)

    def _handle_call_state(self, state: CallState):
        """Handle call state changes"""
        logger.info(f"Call state: {state.value}")
        if self._on_state_change:
            self._on_state_change("call", state.value)

    def _handle_conversation_status(self, status: ConversationStatus):
        """Handle conversation status changes"""
        logger.info(f"Conversation status: {status.value}")
        if self._on_state_change:
            self._on_state_change("conversation", status.value)

    def _save_call_log(self, request: CallRequest, result: ConversationResult,
                       recording_path: Optional[str], duration: float):
        """Save call log to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.CALLS_DIR, f"log_{timestamp}.json")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "phone": request.phone,
            "objective": request.objective,
            "context": request.context,
            "success": result.success,
            "summary": result.summary,
            "collected_info": result.collected_info,
            "transcript": result.transcript,
            "recording_path": recording_path,
            "duration_seconds": duration
        }

        os.makedirs(config.CALLS_DIR, exist_ok=True)

        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Call log saved to {log_path}")
        except Exception as e:
            logger.error(f"Failed to save call log: {e}")

    async def stop(self):
        """Stop the current call"""
        self._is_running = False
        await self.call_controller.end_call()


# CLI interface
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Phone Agent")
    parser.add_argument("phone", help="Phone number to call")
    parser.add_argument("objective", help="What you want to accomplish")
    parser.add_argument("--context", "-c", action="append", nargs=2,
                       metavar=("KEY", "VALUE"), help="Context key-value pairs")

    args = parser.parse_args()

    # Build context dict
    context = {}
    if args.context:
        for key, value in args.context:
            context[key] = value

    # Create agent and make call
    agent = PhoneAgent()

    request = CallRequest(
        phone=args.phone,
        objective=args.objective,
        context=context
    )

    print(f"\n{'='*60}")
    print(f"AI Phone Agent")
    print(f"{'='*60}")
    print(f"Calling: {args.phone}")
    print(f"Objective: {args.objective}")
    if context:
        print(f"Context: {context}")
    print(f"{'='*60}\n")

    result = await agent.call(request)

    print(f"\n{'='*60}")
    print(f"Call Complete")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"\nSummary: {result.summary}")
    print(f"\nCollected Info: {json.dumps(result.collected_info, indent=2)}")
    if result.recording_path:
        print(f"\nRecording: {result.recording_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
