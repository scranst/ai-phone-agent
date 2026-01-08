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


def split_sms(text: str, max_len: int = 155) -> list:
    """Split a long message into SMS-sized chunks, trying to break at word boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Find last space before max_len
        break_point = text.rfind(' ', 0, max_len)
        if break_point == -1:
            # No space found, hard break
            break_point = max_len

        chunks.append(text[:break_point].strip())
        text = text[break_point:].strip()

    return chunks


import database

logger = logging.getLogger(__name__)


@dataclass
class CallRequest:
    phone: str
    objective: str
    context: dict = field(default_factory=dict)
    enable_tools: bool = False  # Enable AI tools (web search, etc.) for this call


@dataclass
class CallResult:
    success: bool
    summary: str
    transcript: list = field(default_factory=list)
    duration_seconds: float = 0.0


class PhoneAgentLocal:
    """Phone agent using local STT/TTS pipeline"""

    def __init__(self, pre_initialize: bool = True, conversation_engine: Optional[LocalConversationEngine] = None, modem: Optional[SIM7600Modem] = None):
        # Use provided modem or create new one
        self.modem = modem if modem else SIM7600Modem()
        self._owns_modem = modem is None  # Track if we created it
        self.audio = AudioRouterSIM7600()

        # Use provided conversation engine or create new one
        if conversation_engine is not None:
            self.conversation = conversation_engine
            self._owns_conversation = False  # Don't re-initialize
        else:
            self.conversation = LocalConversationEngine()
            self._owns_conversation = True

        self._call_active = False
        self._audio_output_queue = asyncio.Queue()

        # External callbacks (for web UI etc)
        self._external_state_callback = None
        self._external_transcript_callback = None

        # Pre-initialize to avoid delay when calling (only if we created the engine)
        if pre_initialize and self._owns_conversation:
            self.conversation.initialize()

    async def _generate_greeting_async(self) -> Optional[bytes]:
        """Generate greeting text and audio in background (runs during ringing)"""
        try:
            # Run the blocking LLM and TTS calls in executor to not block
            loop = asyncio.get_event_loop()

            # Generate greeting text (LLM call - this is the slow part)
            greeting_text = await loop.run_in_executor(
                None, self.conversation.get_initial_greeting
            )

            if not greeting_text:
                return None

            # Synthesize to audio (TTS call - usually fast)
            greeting_audio = await loop.run_in_executor(
                None, self.conversation.synthesize_greeting, greeting_text
            )

            return greeting_audio
        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            return None

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
        lead = None

        # Look up lead by phone number for context
        try:
            lead = database.get_lead_by_phone(request.phone)
            if lead:
                logger.info(f"Found lead: {lead.get('first_name', '')} {lead.get('last_name', '')} at {lead.get('company', 'Unknown')}")
                # Add lead context
                request.context = self._build_lead_context(lead, request.context)
        except Exception as e:
            logger.warning(f"Could not look up lead: {e}")

        try:
            # 1. Connect modem (skip if already connected - e.g., shared modem)
            if not self.modem.dev:
                if not self.modem.connect():
                    raise Exception("Failed to connect to SIM7600 modem")

            # 2. Set up callbacks (conversation engine already initialized in __init__)
            self.conversation.on_state_change(self._on_state_change)
            self.conversation.on_transcript(self._on_transcript)

            # 3. Start audio
            if not self.audio.start():
                raise Exception("Failed to start audio routing")

            # self.audio.start_recording()  # TEMPORARILY DISABLED - not saving WAV files

            # 4. Dial
            if not self.modem.dial(request.phone):
                raise Exception("Failed to dial")

            # 5. WHILE RINGING: Prepare greeting in parallel to save time
            # Start conversation engine and generate greeting while phone rings
            conv_config = ConversationConfig(
                objective=request.objective,
                context=request.context,
                enable_tools=request.enable_tools
            )
            self.conversation.start(conv_config)

            # Generate greeting in background while waiting for answer
            logger.info("Generating greeting while ringing...")
            greeting_task = asyncio.create_task(self._generate_greeting_async())

            # Wait for call to connect
            call_state = await self._wait_for_connect(timeout=60)

            if call_state != CallState.CONNECTED:
                greeting_task.cancel()
                raise Exception("Call did not connect")

            logger.info("Call connected!")
            self._call_active = True

            # Wait for audio to stabilize after connection
            await asyncio.sleep(0.3)

            # Get the pre-generated greeting (should be ready by now)
            greeting_audio = await greeting_task

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

            # Normal cleanup - hang up
            try:
                self.modem.hangup()
            except:
                pass

            # Save recording (TEMPORARILY DISABLED)
            recording_path = None  # self.audio.stop_recording()
            self.audio.stop()

            # Get call result for logging and SMS
            call_result = self.conversation.get_result()

            # Save call log
            self._save_call_log(request, call_result, recording_path)

            # Log interaction to database if we have a lead
            if lead:
                self._log_interaction_to_db(lead, call_result, recording_path)

            # Send SMS summary if enabled (wait for modem to settle after call)
            import time as time_module
            time_module.sleep(2)  # Give modem time to return to command mode
            self._send_sms_summary(request.phone, call_result)

            # Disconnect modem (after SMS is sent) - only if we own it
            # When using a shared modem (passed in constructor), don't disconnect
            if self._owns_modem:
                try:
                    self.modem.disconnect()
                except:
                    pass

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

    def _send_sms_summary(self, called_phone: str, result):
        """Send SMS summary after call if enabled in settings"""
        try:
            # Load settings to check if SMS is enabled
            settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
            if not os.path.exists(settings_path):
                return

            with open(settings_path, 'r') as f:
                settings = json.load(f)

            # Check if SMS is enabled
            if not settings.get("SMS_ENABLED", False):
                logger.info("SMS notifications disabled, skipping")
                return

            # Get callback number to send SMS to
            callback_number = settings.get("CALLBACK_NUMBER", "").replace("-", "").replace(" ", "")
            if not callback_number:
                logger.warning("No callback number configured, skipping SMS")
                return

            # Build SMS message - use full summary, will split if needed
            if result.success:
                summary = result.summary if result.summary else "Call completed successfully"
                message = f"AI Call Summary: {summary}"
            else:
                message = f"AI Call to {called_phone} ended. {result.summary if result.summary else 'No summary available'}"

            # Split into multiple SMS if too long
            chunks = split_sms(message)
            logger.info(f"Sending SMS summary to {callback_number} ({len(chunks)} message(s))")

            all_sent = True
            for i, chunk in enumerate(chunks):
                success = self.modem.send_sms(callback_number, chunk)
                if not success:
                    logger.error(f"Failed to send SMS chunk {i+1}/{len(chunks)}")
                    all_sent = False
                elif len(chunks) > 1:
                    time.sleep(0.5)  # Brief pause between messages

            if all_sent:
                logger.info("SMS summary sent successfully")
                # Save to database for conversation history (save full message)
                try:
                    database.save_message({
                        "channel": "sms",
                        "direction": "outbound",
                        "from_address": "",  # Our modem number
                        "to_address": callback_number,
                        "body": message,
                        "status": "sent",
                        "provider": "modem",
                        "sent_at": datetime.now().isoformat()
                    })
                except Exception as db_err:
                    logger.error(f"Failed to save SMS to database: {db_err}")
            else:
                logger.error("Failed to send some SMS chunks")

        except Exception as e:
            logger.error(f"Error sending SMS summary: {e}")

    def _build_lead_context(self, lead: dict, existing_context: dict) -> dict:
        """Build context from lead data, merged with existing context"""
        lead_context = {}

        # Contact info
        if lead.get('first_name'):
            lead_context['LEAD_FIRST_NAME'] = lead['first_name']
        if lead.get('last_name'):
            lead_context['LEAD_LAST_NAME'] = lead['last_name']
        if lead.get('first_name') or lead.get('last_name'):
            lead_context['LEAD_NAME'] = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
        if lead.get('email'):
            lead_context['LEAD_EMAIL'] = lead['email']
        if lead.get('phone'):
            lead_context['LEAD_PHONE'] = lead['phone']

        # Company info
        if lead.get('company'):
            lead_context['LEAD_COMPANY'] = lead['company']
        if lead.get('title'):
            lead_context['LEAD_TITLE'] = lead['title']
        if lead.get('industry'):
            lead_context['LEAD_INDUSTRY'] = lead['industry']

        # Personalization
        if lead.get('icebreaker'):
            lead_context['ICEBREAKER'] = lead['icebreaker']
        if lead.get('trigger_event'):
            lead_context['TRIGGER_EVENT'] = lead['trigger_event']
        if lead.get('pain_points'):
            lead_context['PAIN_POINTS'] = lead['pain_points']

        # Notes
        if lead.get('notes'):
            lead_context['LEAD_NOTES'] = lead['notes']

        # Status
        if lead.get('status'):
            lead_context['LEAD_STATUS'] = lead['status']
        if lead.get('sentiment_status'):
            lead_context['LEAD_SENTIMENT'] = lead['sentiment_status']

        # Merge: existing context overrides lead context
        return {**lead_context, **existing_context}

    def _log_interaction_to_db(self, lead: dict, result, recording_path: Optional[str]):
        """Log call interaction to database and update lead"""
        try:
            lead_id = lead['id']

            # Determine outcome based on result
            outcome = 'completed'
            if not result.success:
                outcome = 'failed'
            elif hasattr(result, 'summary'):
                summary_lower = result.summary.lower() if result.summary else ''
                if 'booked' in summary_lower or 'scheduled' in summary_lower or 'meeting' in summary_lower:
                    outcome = 'booked'
                elif 'callback' in summary_lower:
                    outcome = 'callback'
                elif 'not interested' in summary_lower or 'no interest' in summary_lower:
                    outcome = 'not_interested'
                elif 'voicemail' in summary_lower:
                    outcome = 'voicemail'
                elif 'no answer' in summary_lower:
                    outcome = 'no_answer'

            # Log interaction
            interaction_data = {
                'channel': 'call',
                'direction': 'outbound',
                'duration_seconds': int(result.duration_seconds) if hasattr(result, 'duration_seconds') else 0,
                'recording_path': recording_path,
                'transcript': json.dumps(result.transcript) if hasattr(result, 'transcript') else None,
                'summary': result.summary if hasattr(result, 'summary') else None,
                'outcome': outcome
            }

            database.log_interaction(lead_id, interaction_data)
            logger.info(f"Logged interaction for lead {lead_id}")

            # Update lead status based on outcome
            lead_updates = {'last_contacted_at': datetime.now().isoformat()}

            if outcome == 'booked':
                lead_updates['status'] = 'MEETING_BOOKED'
            elif outcome == 'not_interested':
                lead_updates['status'] = 'LOST'
            elif outcome in ['completed', 'callback']:
                # Only upgrade from NEW to CONTACTED
                if lead.get('status') == 'NEW':
                    lead_updates['status'] = 'CONTACTED'

            database.update_lead(lead_id, lead_updates)
            logger.info(f"Updated lead {lead_id} status")

        except Exception as e:
            logger.error(f"Error logging interaction to database: {e}")


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
