"""
Incoming Call Handler using Local STT/TTS

Uses:
- SIM7600 modem for calls
- Local Whisper for speech recognition
- Claude Haiku for conversation
- Local Piper for text-to-speech

Waits for incoming calls and handles them with a configurable persona.
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import json
import os

from sim7600_modem import SIM7600Modem, CallState
from audio_router_sim7600 import AudioRouterSIM7600
from conversation_local import LocalConversationEngine, ConversationConfig, ConversationState

import config

logger = logging.getLogger(__name__)


@dataclass
class IncomingCallResult:
    success: bool
    caller_id: str
    summary: str
    transcript: list = field(default_factory=list)
    duration_seconds: float = 0.0


class IncomingCallHandler:
    """Handler for incoming calls using local STT/TTS pipeline"""

    def __init__(self):
        self.modem = SIM7600Modem()
        self.audio = AudioRouterSIM7600()
        self.conversation = LocalConversationEngine()

        self._call_active = False
        self._listening = False
        self._audio_output_queue = asyncio.Queue()

        # External callbacks (for web UI etc)
        self._external_state_callback = None
        self._external_transcript_callback = None
        self._incoming_call_callback = None

    def on_state_change(self, callback: Callable):
        """Register external callback for state changes"""
        self._external_state_callback = callback

    def on_transcript(self, callback: Callable):
        """Register external callback for transcript updates"""
        self._external_transcript_callback = callback

    def on_incoming_call(self, callback: Callable):
        """Register callback for when an incoming call is detected"""
        self._incoming_call_callback = callback

    def _load_incoming_settings(self) -> dict:
        """Load incoming call settings from settings.json"""
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        if not os.path.exists(settings_path):
            return {}

        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            incoming = settings.get("incoming", {})

            # Substitute placeholders like {MY_NAME}
            persona = incoming.get("PERSONA", "")
            greeting = incoming.get("GREETING", "")

            for key, value in settings.items():
                if isinstance(value, str):
                    placeholder = "{" + key + "}"
                    persona = persona.replace(placeholder, value)
                    greeting = greeting.replace(placeholder, value)

            return {
                "enabled": incoming.get("ENABLED", False),
                "persona": persona,
                "greeting": greeting,
                "my_name": settings.get("MY_NAME", ""),
                "callback_number": settings.get("CALLBACK_NUMBER", ""),
                "settings": settings
            }
        except Exception as e:
            logger.error(f"Error loading incoming settings: {e}")
            return {}

    async def start_listening(self):
        """Start listening for incoming calls"""
        logger.info("Starting incoming call listener")

        settings = self._load_incoming_settings()
        if not settings.get("enabled", False):
            logger.info("Incoming calls disabled in settings")
            return

        self._listening = True

        try:
            # Connect modem
            if not self.modem.connect():
                raise Exception("Failed to connect to SIM7600 modem")

            logger.info("Modem connected, waiting for incoming calls...")

            while self._listening:
                # Wait for incoming call (5 second timeout to check listening flag)
                caller_id = self.modem.wait_for_incoming_call(timeout=5)

                if caller_id:
                    logger.info(f"Incoming call from: {caller_id}")

                    # Notify callback
                    if self._incoming_call_callback:
                        self._incoming_call_callback(caller_id)

                    # Reload settings in case they changed
                    settings = self._load_incoming_settings()

                    if settings.get("enabled", False):
                        # Handle the call
                        result = await self._handle_incoming_call(caller_id, settings)
                        logger.info(f"Call ended: {result.summary}")
                    else:
                        logger.info("Incoming calls disabled, rejecting")
                        self.modem.reject_call()

        except Exception as e:
            logger.error(f"Incoming call listener error: {e}")
        finally:
            self._listening = False
            try:
                self.modem.disconnect()
            except:
                pass

    def stop_listening(self):
        """Stop listening for incoming calls"""
        logger.info("Stopping incoming call listener")
        self._listening = False

    async def _handle_incoming_call(self, caller_id: str, settings: dict) -> IncomingCallResult:
        """Handle an incoming call"""
        logger.info(f"Handling incoming call from {caller_id}")
        start_time = time.time()
        recording_path = None

        try:
            # Initialize conversation engine
            self.conversation.initialize()

            # Set up callbacks
            self.conversation.on_state_change(self._on_state_change)
            self.conversation.on_transcript(self._on_transcript)

            # Start audio
            if not self.audio.start():
                raise Exception("Failed to start audio routing")

            self.audio.start_recording()

            # Answer the call
            if not self.modem.answer():
                raise Exception("Failed to answer call")

            logger.info("Call answered!")
            self._call_active = True

            # Wait for audio to stabilize
            await asyncio.sleep(0.5)

            # Start conversation with incoming call objective
            persona = settings.get("persona", "You are a helpful assistant.")
            greeting = settings.get("greeting", "Hello, how can I help you?")

            conv_config = ConversationConfig(
                objective=f"Answer this incoming call professionally. {persona}",
                context={
                    "caller_id": caller_id,
                    "my_name": settings.get("my_name", ""),
                    "callback_number": settings.get("callback_number", ""),
                    "direction": "incoming"
                }
            )
            self.conversation.start(conv_config)

            # Speak the greeting
            logger.info(f"Speaking greeting: {greeting}")
            greeting_audio = self.conversation.synthesize_greeting(greeting)

            if greeting_audio:
                self.conversation.set_speaking(True)
                await self.audio.write_audio_async(greeting_audio)
                await asyncio.sleep(0.3)
                self.conversation.set_speaking(False)

            # Clear buffered audio
            self.audio.clear_input_buffer()

            # Main conversation loop
            await self._run_conversation_loop()

            # Get result
            result = self.conversation.get_result()

            return IncomingCallResult(
                success=result.success,
                caller_id=caller_id,
                summary=result.summary,
                transcript=result.transcript,
                duration_seconds=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Incoming call failed: {e}")
            return IncomingCallResult(
                success=False,
                caller_id=caller_id,
                summary=str(e),
                transcript=[],
                duration_seconds=time.time() - start_time
            )

        finally:
            # Cleanup
            self._call_active = False
            self.conversation.stop()

            try:
                self.modem.hangup()
            except:
                pass

            # Save recording
            recording_path = self.audio.stop_recording()
            self.audio.stop()

            # Get call result for logging
            call_result = self.conversation.get_result()

            # Save call log
            self._save_call_log(caller_id, call_result, recording_path, settings)

            # Send SMS summary if enabled
            time.sleep(2)  # Give modem time to return to command mode
            self._send_sms_summary(caller_id, call_result, settings.get("settings", {}))

    async def _run_conversation_loop(self):
        """Main conversation loop"""
        logger.info("Starting conversation loop")

        # Create tasks
        audio_task = asyncio.create_task(self._audio_input_loop())
        output_task = asyncio.create_task(self._audio_output_loop())
        monitor_task = asyncio.create_task(self._monitor_call())

        try:
            await asyncio.gather(audio_task, output_task, monitor_task)
        except asyncio.CancelledError:
            pass
        finally:
            audio_task.cancel()
            output_task.cancel()
            monitor_task.cancel()

    async def _audio_input_loop(self):
        """Read audio from phone and process"""
        while self._call_active:
            audio_bytes = await self.audio.read_audio()

            if audio_bytes:
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                response_audio = self.conversation.process_audio(audio_chunk)

                if response_audio:
                    await self._audio_output_queue.put(response_audio)

            await asyncio.sleep(0.01)

    async def _audio_output_loop(self):
        """Play audio responses to phone"""
        while self._call_active:
            try:
                response_audio = await asyncio.wait_for(
                    self._audio_output_queue.get(),
                    timeout=0.1
                )

                if response_audio:
                    logger.info(f"Playing response audio ({len(response_audio)} bytes)")

                    self.conversation.set_speaking(True)
                    await self.audio.write_audio_async(response_audio)
                    await asyncio.sleep(0.3)
                    self.conversation.set_speaking(False)

                    self.audio.clear_input_buffer()

                    if self.conversation.state == ConversationState.COMPLETED:
                        logger.info("Conversation completed")
                        await asyncio.sleep(1)
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
        if self._external_state_callback:
            self._external_state_callback(state)

    def _on_transcript(self, role: str, text: str):
        """Handle transcript updates"""
        logger.info(f"{role.title()}: {text}")
        if self._external_transcript_callback:
            self._external_transcript_callback(role, text)

    def _save_call_log(self, caller_id: str, result, recording_path: Optional[str], settings: dict):
        """Save call log to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "phone": caller_id,
            "direction": "incoming",
            "objective": "Incoming call",
            "context": {"caller_id": caller_id},
            "success": result.success,
            "summary": result.summary,
            "transcript": result.transcript,
            "recording_path": recording_path,
            "duration_seconds": result.duration_seconds,
            "engine": "local"
        }

        log_path = os.path.join(
            config.CALLS_DIR,
            f"incoming_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        os.makedirs(config.CALLS_DIR, exist_ok=True)

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Call log saved to {log_path}")

    def _send_sms_summary(self, caller_id: str, result, settings: dict):
        """Send SMS summary after call if enabled in settings"""
        try:
            if not settings.get("SMS_ENABLED", False):
                logger.info("SMS notifications disabled, skipping")
                return

            callback_number = settings.get("CALLBACK_NUMBER", "").replace("-", "").replace(" ", "")
            if not callback_number:
                logger.warning("No callback number configured, skipping SMS")
                return

            # Build SMS message
            if result.success:
                summary = result.summary[:100] if result.summary else "Call completed"
                message = f"Incoming call from {caller_id}: {summary}"
            else:
                message = f"Incoming call from {caller_id} ended. {result.summary[:80] if result.summary else ''}"

            if len(message) > 160:
                message = message[:157] + "..."

            logger.info(f"Sending SMS summary to {callback_number}")
            success = self.modem.send_sms(callback_number, message)

            if success:
                logger.info("SMS summary sent successfully")
            else:
                logger.error("Failed to send SMS summary")

        except Exception as e:
            logger.error(f"Error sending SMS summary: {e}")


# Main entry point for testing
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = IncomingCallHandler()

    def on_incoming(caller_id):
        print(f"\n*** INCOMING CALL FROM: {caller_id} ***\n")

    handler.on_incoming_call(on_incoming)

    print("\n" + "=" * 60)
    print("Incoming Call Handler")
    print("=" * 60)
    print("\nWaiting for incoming calls...")
    print("Press Ctrl+C to stop\n")

    try:
        await handler.start_listening()
    except KeyboardInterrupt:
        print("\nStopping...")
        handler.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
