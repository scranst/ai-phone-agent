"""
Conversation Engine - Manages AI conversation via OpenAI Realtime API

Handles:
- WebSocket connection to OpenAI Realtime API
- Audio streaming in/out
- Conversation state and objective tracking
- Hold/transfer detection
"""

import asyncio
import json
import logging
import base64
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import websockets

import config

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class ConversationStatus(Enum):
    INITIALIZING = "initializing"
    LISTENING = "listening"  # Waiting for human to speak
    PROCESSING = "processing"  # AI is thinking
    SPEAKING = "speaking"  # AI is talking
    ON_HOLD = "on_hold"  # Detected hold music
    COMPLETED = "completed"  # Objective achieved
    FAILED = "failed"


@dataclass
class ConversationResult:
    success: bool
    summary: str
    collected_info: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ConversationConfig:
    objective: str
    context: dict
    voice: str = "alloy"
    max_duration: int = 600  # 10 minutes


class ConversationEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.status = ConversationStatus.INITIALIZING
        self.config: Optional[ConversationConfig] = None
        self.transcript: list[dict] = []
        self.collected_info: dict = {}
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._status_callback: Optional[Callable[[ConversationStatus], None]] = None
        self._transcript_callback: Optional[Callable[[str, str], None]] = None
        self._running = False
        self._first_user_speech = False  # Track if user has spoken first
        self._response_in_progress = False  # Track if AI is currently responding
        self._conversation_started = False  # Track if greeting detected and convo started
        self._user_speech_count = 0  # Count user speech events to handle fallback

        # Greeting words to detect when someone answers
        # Include common transcription variants of "hello"
        self.GREETING_WORDS = [
            "hello", "hi", "hey", "yes", "yeah", "speaking",
            "this is", "good morning", "good afternoon", "good evening",
            "how can i help", "may i help", "what can i",
            # Common transcription variants
            "hallo", "ello", "lo", "yo", "you", "yep", "okay",
            "what's up", "sup", "go ahead", "who is this", "can i help"
        ]

    def on_audio_output(self, callback: Callable[[bytes], None]):
        """Register callback for AI audio output"""
        self._audio_callback = callback

    def on_status_change(self, callback: Callable[[ConversationStatus], None]):
        """Register callback for status changes"""
        self._status_callback = callback

    def on_transcript(self, callback: Callable[[str, str], None]):
        """Register callback for transcript updates (role, text)"""
        self._transcript_callback = callback

    def _set_status(self, status: ConversationStatus):
        self.status = status
        if self._status_callback:
            self._status_callback(status)

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on objective and context"""
        context_str = "\n".join(
            f"- {k}: {v}" for k, v in self.config.context.items()
        )

        return f"""You are an AI assistant making a phone call on behalf of a user.

YOUR OBJECTIVE:
{self.config.objective}

CONTEXT/INFORMATION YOU HAVE:
{context_str}

CRITICAL RULES:
1. WAIT for them to speak first - they will say "hello" when they answer.
2. Introduce yourself ONLY ONCE at the start of the call. After that, just respond normally.
3. If you hear ringing, dial tones, or weird sounds - STAY SILENT and wait.
4. Never repeat your introduction. If you've already introduced yourself, just continue the conversation.

IMPORTANT GUIDELINES:
1. WAIT for them to speak first (they will say hello)
2. Be polite, professional, and natural-sounding
3. Speak clearly and at a normal pace
4. If asked who you are, say you're calling on behalf of {self.config.context.get('name', 'the customer')}
5. If put on hold, wait patiently and say "okay, I'll hold"
6. If transferred, re-introduce yourself and your purpose
7. Collect all relevant information (prices, times, confirmation numbers, etc.)
8. When the objective is complete, politely end the call
9. If you cannot complete the objective, politely explain and end the call

DETECTING COMPLETION:
- When you have successfully achieved the objective (got a quote, made an appointment, placed an order, etc.),
  respond with the phrase "OBJECTIVE_COMPLETE" somewhere in your final response before saying goodbye.
- Include all collected information in your response.

Remember: You are on a PHONE CALL. Keep responses conversational and brief. Don't use bullet points or
formatting - just speak naturally. LET THEM SPEAK FIRST."""

    async def connect(self, conversation_config: ConversationConfig) -> bool:
        """Connect to OpenAI Realtime API and configure the session"""
        self.config = conversation_config

        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1")
        ]

        url = f"{OPENAI_REALTIME_URL}?model={config.REALTIME_MODEL}"

        try:
            self.ws = await websockets.connect(url, additional_headers=headers)
            logger.info("Connected to OpenAI Realtime API")

            # Configure the session
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self._build_system_prompt(),
                    "voice": self.config.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.4,  # Lower threshold - more sensitive
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 400,
                        "create_response": False  # Don't auto-create response - we trigger manually
                    }
                }
            }

            await self.ws.send(json.dumps(session_config))
            self._set_status(ConversationStatus.LISTENING)
            self._running = True

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            self._set_status(ConversationStatus.FAILED)
            return False

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to OpenAI"""
        if not self.ws or not self._running:
            return

        # Convert to base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def process_messages(self):
        """Process incoming messages from OpenAI"""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                if not self._running:
                    break

                data = json.loads(message)
                await self._handle_message(data)

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
        finally:
            self._running = False

    async def _handle_message(self, data: dict):
        """Handle a message from OpenAI Realtime API"""
        msg_type = data.get("type", "")

        if msg_type == "session.created":
            logger.info("Session created successfully")

        elif msg_type == "session.updated":
            logger.info("Session updated")

        elif msg_type == "input_audio_buffer.speech_started":
            logger.info(">>> User started speaking (VAD detected)")

        elif msg_type == "input_audio_buffer.speech_stopped":
            logger.info(">>> User stopped speaking (VAD detected)")

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # User's speech was transcribed
            transcript = data.get("transcript", "").strip()
            if not transcript:
                logger.warning("Empty transcript received (speech detected but nothing recognized)")
                # Still count it as a speech event for fallback trigger
                self._user_speech_count += 1
                # If we get an empty transcript, the person likely answered
                # Start the conversation immediately to avoid awkward silence
                if not self._conversation_started:
                    logger.info(f"Fallback: Empty transcript but speech detected - starting conversation")
                    self._conversation_started = True
                    self._response_in_progress = True
                    await self.ws.send(json.dumps({"type": "response.create"}))
                return
            if transcript:
                self.transcript.append({"role": "user", "content": transcript})
                self._user_speech_count += 1

                if self._transcript_callback:
                    self._transcript_callback("user", transcript)
                logger.info(f"User: {transcript}")

                # SIMPLIFIED: Any speech after call connects means someone answered
                # Just respond immediately to any transcribed speech
                if not self._response_in_progress:
                    if not self._conversation_started:
                        logger.info(f"Speech detected: '{transcript}' - starting conversation immediately")
                        self._conversation_started = True

                    self._response_in_progress = True
                    await self.ws.send(json.dumps({"type": "response.create"}))

        elif msg_type == "response.audio.delta":
            # AI audio chunk
            audio_b64 = data.get("delta", "")
            if audio_b64 and self._audio_callback:
                audio_bytes = base64.b64decode(audio_b64)
                self._audio_callback(audio_bytes)
            self._set_status(ConversationStatus.SPEAKING)

        elif msg_type == "response.audio.done":
            # AI finished speaking
            self._set_status(ConversationStatus.LISTENING)

        elif msg_type == "response.audio_transcript.delta":
            # Partial transcript of AI speech
            pass

        elif msg_type == "response.audio_transcript.done":
            # Complete transcript of AI speech
            transcript = data.get("transcript", "")
            if transcript:
                self.transcript.append({"role": "assistant", "content": transcript})
                if self._transcript_callback:
                    self._transcript_callback("assistant", transcript)
                logger.info(f"Assistant: {transcript}")

                # Check for objective completion
                if "OBJECTIVE_COMPLETE" in transcript.upper():
                    self._set_status(ConversationStatus.COMPLETED)
                    # Extract collected info from the response
                    self._parse_collected_info(transcript)

        elif msg_type == "response.done":
            # Response complete - allow new responses to be triggered
            self._response_in_progress = False
            if self.status != ConversationStatus.COMPLETED:
                self._set_status(ConversationStatus.LISTENING)

        elif msg_type == "error":
            error = data.get("error", {})
            logger.error(f"API Error: {error}")

        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    def _parse_collected_info(self, text: str):
        """Extract structured info from AI's completion message"""
        # Simple extraction - in practice could use another LLM call
        info = {}

        # Look for common patterns
        import re

        # Price/cost
        price_match = re.search(r'\$[\d,]+\.?\d*', text)
        if price_match:
            info['price'] = price_match.group()

        # Time/date
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'(?:tomorrow|today|next week)'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info.setdefault('schedule', []).append(match.group())

        # Confirmation number
        conf_match = re.search(r'confirmation\s*(?:number|#)?[:\s]*([A-Z0-9-]+)', text, re.IGNORECASE)
        if conf_match:
            info['confirmation'] = conf_match.group(1)

        self.collected_info = info

    async def trigger_response(self, text: Optional[str] = None):
        """Trigger AI to respond (optionally with a text prompt)"""
        if not self.ws:
            return

        if text:
            # Send text message
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            }
            await self.ws.send(json.dumps(message))

        # Request response
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def start_conversation(self):
        """Wait for callee to speak first - they say hello, then AI responds"""
        # Don't trigger automatic greeting - let the callee speak first
        # The AI's instructions tell it to introduce itself when someone speaks
        pass

    async def disconnect(self):
        """Disconnect from the API"""
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("Disconnected from Realtime API")

    def get_result(self) -> ConversationResult:
        """Get the conversation result"""
        # Generate summary from transcript
        summary = ""
        if self.transcript:
            # Last assistant message often contains summary
            for msg in reversed(self.transcript):
                if msg["role"] == "assistant":
                    summary = msg["content"]
                    break

        return ConversationResult(
            success=self.status == ConversationStatus.COMPLETED,
            summary=summary,
            collected_info=self.collected_info,
            transcript=self.transcript
        )


# Test function
async def test_conversation():
    engine = ConversationEngine()

    config = ConversationConfig(
        objective="Order a large pepperoni pizza for delivery",
        context={
            "name": "Scott",
            "address": "123 Main St",
            "phone": "555-1234"
        }
    )

    print("System prompt would be:")
    print("-" * 60)
    engine.config = config
    print(engine._build_system_prompt())
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(test_conversation())
