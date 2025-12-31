"""
AI Phone Agent for SIM7600 HAT - Direct cellular calls via USB modem

Uses:
- SIM7600G-H HAT (B) for cellular voice calls
- 3.5mm audio jack for call audio (via USB sound card)
- OpenAI Realtime API for AI conversation
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from sim7600_modem import SIM7600Modem, CallState, CallInfo
from audio_router_sim7600 import AudioRouterSIM7600
from conversation import ConversationEngine, ConversationConfig, ConversationStatus, ConversationResult
from ringback_detector import RingbackDetector
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


class PhoneAgentSIM7600:
    """AI Phone Agent using SIM7600 cellular modem"""

    def __init__(self,
                 audio_input_device: Optional[str] = None,
                 audio_output_device: Optional[str] = None):
        """
        Initialize the phone agent.

        Args:
            audio_input_device: Name of device receiving HAT speaker output
            audio_output_device: Name of device sending to HAT mic input
        """
        self.modem = SIM7600Modem()
        # Use SIM7600 audio config by default
        self.audio_router = AudioRouterSIM7600(
            input_device_name=audio_input_device or config.SIM7600_AUDIO_INPUT,
            output_device_name=audio_output_device or config.SIM7600_AUDIO_OUTPUT
        )
        self.conversation: Optional[ConversationEngine] = None
        self._current_request: Optional[CallRequest] = None
        self._is_running = False

        # Event callbacks for UI
        self._on_state_change: Optional[callable] = None
        self._on_transcript: Optional[callable] = None

    def on_state_change(self, callback):
        """Register callback for state changes"""
        self._on_state_change = callback

    def on_transcript(self, callback):
        """Register callback for transcript updates"""
        self._on_transcript = callback

    def connect_modem(self) -> bool:
        """Connect to the SIM7600 modem"""
        return self.modem.connect()

    def disconnect_modem(self):
        """Disconnect from the modem"""
        self.modem.disconnect()

    def get_signal_strength(self) -> Optional[int]:
        """Get cellular signal strength (0-31, 99=unknown)"""
        response = self.modem._send_at("AT+CSQ")
        try:
            # Parse +CSQ: rssi,ber
            if "+CSQ:" in response:
                parts = response.split(":")[1].split(",")
                return int(parts[0].strip())
        except:
            pass
        return None

    def get_network_info(self) -> dict:
        """Get network registration info"""
        info = {}

        # Operator name
        response = self.modem._send_at("AT+COPS?")
        if "+COPS:" in response:
            try:
                parts = response.split('"')
                if len(parts) >= 2:
                    info["operator"] = parts[1]
            except:
                pass

        # Signal strength
        signal = self.get_signal_strength()
        if signal is not None:
            info["signal_rssi"] = signal
            # Convert to dBm approximately
            if signal < 99:
                info["signal_dbm"] = -113 + (signal * 2)

        return info

    async def call(self, request: CallRequest) -> CallResult:
        """
        Make an AI-powered phone call via SIM7600.

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

        try:
            # Ensure modem is connected
            if not self.modem.dev:
                if not self.modem.connect():
                    raise Exception("Failed to connect to SIM7600 modem")

            # Initialize conversation engine
            self.conversation = ConversationEngine()

            # Set up callbacks
            self.conversation.on_audio_output(self._handle_ai_audio)
            self.conversation.on_transcript(self._handle_transcript)
            self.conversation.on_status_change(self._handle_conversation_status)

            # Set up modem state callbacks
            self.modem.on_state_change(self._handle_call_state)

            # Start audio routing
            if not self.audio_router.start():
                raise Exception("Failed to start audio routing - check audio device configuration")

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

            # Dial the phone number
            if not self.modem.dial(request.phone):
                raise Exception("Failed to initiate call")

            # Wait for call to connect (modem reports connected early)
            await self._wait_for_call_connect()

            logger.info("Call connected - starting audio processing immediately")

            # Start processing audio IMMEDIATELY (sends to OpenAI)
            # The greeting detection in conversation.py will handle when to respond
            audio_task = asyncio.create_task(self._process_audio())
            message_task = asyncio.create_task(self.conversation.process_messages())

            # Very brief delay to let audio buffers stabilize
            # The greeting detection will handle when to respond
            await asyncio.sleep(1)

            logger.info("AI ready - waiting for greeting to respond")

            # AI will respond when it hears greeting (via transcript trigger)
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
            self.modem.hangup()

            # Stop recording and save
            recording_path = self.audio_router.stop_recording()

            # Disconnect from OpenAI
            if self.conversation:
                await self.conversation.disconnect()

            # Stop audio routing
            self.audio_router.stop()

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

    async def _wait_for_answer(self, timeout: float = 60):
        """
        Wait for call to be answered using ringback tone detection.

        Uses Goertzel algorithm to detect 440Hz + 480Hz ringback tones.
        When ringback stops and voice is detected, the call is answered.
        """
        import numpy as np

        detector = RingbackDetector(sample_rate=24000)
        start = asyncio.get_event_loop().time()

        # Track ringback state
        ringback_detected = False
        voice_detected_count = 0
        required_voice_count = 3  # Need 3 consecutive voice detections

        logger.info("Listening for ringback tones (440Hz + 480Hz)...")

        while self._is_running:
            elapsed = asyncio.get_event_loop().time() - start

            # Check timeout
            if elapsed > timeout:
                logger.warning("Answer detection timeout - starting anyway")
                break

            # Check if modem reports call ended
            call_info = self.modem.get_call_info()
            if call_info and call_info.state == CallState.ENDED:
                logger.warning("Call ended before answer")
                break

            # Read audio chunk
            try:
                audio_bytes = self.audio_router._input_queue.get_nowait()

                # Process with ringback detector
                result = detector.process_audio(audio_bytes)

                if result["is_ringback"]:
                    ringback_detected = True
                    voice_detected_count = 0
                    if elapsed > 2:  # Only log after initial period
                        logger.debug(f"Ringback tone detected (elapsed: {elapsed:.1f}s)")

                elif result["is_voice"] and elapsed > 3:
                    # Voice detected after at least 3 seconds
                    voice_detected_count += 1
                    logger.debug(f"Voice detected: {voice_detected_count}/{required_voice_count}")

                    if voice_detected_count >= required_voice_count:
                        if ringback_detected:
                            logger.info(f"Call answered! (ringback stopped, voice detected at {elapsed:.1f}s)")
                        else:
                            logger.info(f"Voice detected at {elapsed:.1f}s (no ringback heard)")
                        return

                # Also check if detector flagged as answered
                if result["answered"]:
                    logger.info(f"Answer detected by ringback detector at {elapsed:.1f}s")
                    return

            except:
                pass

            await asyncio.sleep(0.02)  # 20ms loop

        logger.info("Proceeding with conversation")

    async def _wait_for_call_connect(self, timeout: float = 60):
        """Wait for the call to be answered"""
        start = asyncio.get_event_loop().time()

        while self._is_running:
            call_info = self.modem.get_call_info()

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
            call_info = self.modem.get_call_info()
            conv_status = self.conversation.status if self.conversation else None

            # Check if call ended
            if call_info and call_info.state == CallState.ENDED:
                logger.info("Call ended")
                break

            # Check if objective completed
            if conv_status == ConversationStatus.COMPLETED:
                logger.info("Objective completed")
                await asyncio.sleep(3)  # Brief pause for goodbye
                break

            # Check timeout
            if call_info and call_info.connect_time:
                duration = asyncio.get_event_loop().time() - call_info.connect_time
                if duration > config.MAX_CALL_DURATION:
                    logger.warning("Call duration limit reached")
                    break

            await asyncio.sleep(0.5)

    async def _process_audio(self):
        """Process audio between SIM7600 and OpenAI"""
        audio_chunks_sent = 0
        while self._is_running:
            # Read audio from modem (via USB sound card)
            audio = await self.audio_router.read_audio()

            if audio and self.conversation:
                # Send to OpenAI
                await self.conversation.send_audio(audio)
                audio_chunks_sent += 1
                if audio_chunks_sent == 1:
                    logger.info("First audio chunk sent to OpenAI")
                elif audio_chunks_sent % 200 == 0:
                    logger.info(f"Audio chunks sent: {audio_chunks_sent}")

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
            "duration_seconds": duration,
            "modem": "SIM7600G-H"
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
        self.modem.hangup()


# CLI interface
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Phone Agent (SIM7600)")
    parser.add_argument("phone", nargs="?", help="Phone number to call")
    parser.add_argument("objective", nargs="?", help="What you want to accomplish")
    parser.add_argument("--context", "-c", action="append", nargs=2,
                       metavar=("KEY", "VALUE"), help="Context key-value pairs")
    parser.add_argument("--test", action="store_true", help="Test modem connection only")
    parser.add_argument("--audio-in", help="Audio input device name")
    parser.add_argument("--audio-out", help="Audio output device name")

    args = parser.parse_args()

    # Create agent
    agent = PhoneAgentSIM7600(
        audio_input_device=args.audio_in,
        audio_output_device=args.audio_out
    )

    print(f"\n{'='*60}")
    print("AI Phone Agent (SIM7600)")
    print(f"{'='*60}")

    # Connect to modem
    print("\nConnecting to SIM7600 modem...")
    if not agent.connect_modem():
        print("ERROR: Failed to connect to modem")
        print("Make sure the SIM7600 HAT is connected and has power")
        return

    # Show network info
    network_info = agent.get_network_info()
    if network_info:
        print(f"\nNetwork: {network_info.get('operator', 'Unknown')}")
        signal = network_info.get('signal_dbm')
        if signal:
            print(f"Signal: {signal} dBm")

    if args.test:
        print("\nModem test successful!")
        print("Ready to make calls.")
        agent.disconnect_modem()
        return

    if not args.phone or not args.objective:
        print("\nUsage: python agent_sim7600.py <phone> <objective>")
        print("Example: python agent_sim7600.py '+15551234567' 'Schedule an appointment'")
        agent.disconnect_modem()
        return

    # Build context dict
    context = {}
    if args.context:
        for key, value in args.context:
            context[key] = value

    request = CallRequest(
        phone=args.phone,
        objective=args.objective,
        context=context
    )

    print(f"\nCalling: {args.phone}")
    print(f"Objective: {args.objective}")
    if context:
        print(f"Context: {context}")
    print(f"{'='*60}\n")

    # Make the call
    result = await agent.call(request)

    print(f"\n{'='*60}")
    print("Call Complete")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"\nSummary: {result.summary}")
    print(f"\nCollected Info: {json.dumps(result.collected_info, indent=2)}")
    if result.recording_path:
        print(f"\nRecording: {result.recording_path}")
    print(f"{'='*60}\n")

    # Cleanup
    agent.disconnect_modem()


if __name__ == "__main__":
    asyncio.run(main())
