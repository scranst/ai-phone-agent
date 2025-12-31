"""
Local Conversation Engine

Orchestrates the STT -> LLM -> TTS pipeline for phone conversations.
Uses local Whisper for STT, Claude Haiku for LLM, and Piper for TTS.
"""

import asyncio
import logging
import re
import time
import numpy as np
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass, field

from vad import VoiceActivityDetector
from stt import SpeechToText
from tts import TextToSpeech
from llm import LLMEngine

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    FAILED = "failed"
    TRANSFERRING = "transferring"


@dataclass
class ConversationResult:
    success: bool
    summary: str
    transcript: list = field(default_factory=list)
    duration_seconds: float = 0.0
    transfer_to: str = None  # Phone number to transfer to


@dataclass
class ConversationConfig:
    objective: str
    context: dict
    max_duration: int = 300  # 5 minutes


class LocalConversationEngine:
    """
    Manages phone conversation using local STT/TTS.

    Flow:
    1. Listen for speech (VAD)
    2. When speech ends, transcribe (Whisper)
    3. Generate response (Claude Haiku)
    4. Synthesize speech (Piper)
    5. Play audio
    6. Go back to step 1
    """

    def __init__(self):
        # Components
        self.vad: Optional[VoiceActivityDetector] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.llm: Optional[LLMEngine] = None

        # State
        self.state = ConversationState.IDLE
        self.config: Optional[ConversationConfig] = None
        self.transcript: list = []
        self.start_time: float = 0

        # Callbacks
        self._audio_output_callback: Optional[Callable[[bytes], None]] = None
        self._state_callback: Optional[Callable[[ConversationState], None]] = None
        self._transcript_callback: Optional[Callable[[str, str], None]] = None

        # Control flags
        self._running = False
        self._is_speaking = False  # True while playing TTS audio
        self._greeting_sent = False
        self._transfer_number = None  # Number to transfer to

        # Audio sample rate
        self.sample_rate = 24000

    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing local conversation engine...")

        self.vad = VoiceActivityDetector(
            aggressiveness=3,  # 0-3, higher = more aggressive filtering
            sample_rate=self.sample_rate,
            min_speech_ms=200,
            min_silence_ms=500,  # Faster turn-taking
            max_speech_ms=30000,  # 30 second max
            energy_threshold=4000  # Higher threshold to filter connection noise (real speech ~7000-12000 RMS)
        )

        self.stt = SpeechToText(
            model_size="base.en",
            device="cpu",
            compute_type="int8"
        )

        self.tts = TextToSpeech(
            output_sample_rate=self.sample_rate
        )

        self.llm = LLMEngine()

        logger.info("Local conversation engine initialized")

    def on_audio_output(self, callback: Callable[[bytes], None]):
        """Register callback for audio output"""
        self._audio_output_callback = callback

    def on_state_change(self, callback: Callable[[ConversationState], None]):
        """Register callback for state changes"""
        self._state_callback = callback

    def on_transcript(self, callback: Callable[[str, str], None]):
        """Register callback for transcript updates"""
        self._transcript_callback = callback

    def _set_state(self, state: ConversationState):
        """Update state and notify callback"""
        self.state = state
        if self._state_callback:
            self._state_callback(state)

    def start(self, config: ConversationConfig):
        """Start the conversation"""
        self.config = config
        self.transcript = []
        self.start_time = time.time()
        self._running = True
        self._greeting_sent = False

        # Reset VAD state
        if self.vad:
            self.vad.reset()

        # Set LLM objective
        self.llm.set_objective(config.objective, config.context)

        self._set_state(ConversationState.LISTENING)
        logger.info("Conversation started - listening for speech")

    def stop(self):
        """Stop the conversation"""
        self._running = False
        self._set_state(ConversationState.IDLE)

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[bytes]:
        """
        Process incoming audio chunk.

        Args:
            audio_chunk: Audio samples as int16 numpy array

        Returns:
            Audio response bytes to play, or None
        """
        if not self._running:
            return None

        # Don't process audio while we're speaking (echo suppression)
        if self._is_speaking:
            return None

        # Check max duration
        if time.time() - self.start_time > self.config.max_duration:
            logger.info("Max duration reached")
            self._set_state(ConversationState.COMPLETED)
            self._running = False
            return None

        # Process through VAD
        vad_result = self.vad.process_chunk(audio_chunk)

        if vad_result['speech_started']:
            logger.info(">>> Speech started")

        if vad_result['speech_ended'] and vad_result['audio_buffer'] is not None:
            logger.info(">>> Speech ended - processing")
            return self._process_utterance(vad_result['audio_buffer'])

        return None

    def _process_utterance(self, audio_buffer: np.ndarray) -> Optional[bytes]:
        """
        Process a complete utterance.

        Args:
            audio_buffer: Complete speech audio

        Returns:
            Response audio bytes
        """
        # Skip if audio buffer is too quiet (likely noise, not speech)
        rms = np.sqrt(np.mean(audio_buffer.astype(np.float32) ** 2))
        if rms < 3000:
            logger.info(f"Skipping low-energy audio (RMS={rms:.0f})")
            self._set_state(ConversationState.LISTENING)
            return None

        self._set_state(ConversationState.PROCESSING)

        # 1. Transcribe
        logger.info(f"Transcribing (RMS={rms:.0f})...")
        user_text = self.stt.transcribe(audio_buffer, self.sample_rate)

        if not user_text.strip():
            logger.warning("Empty transcription")
            self._set_state(ConversationState.LISTENING)
            return None

        logger.info(f"User: {user_text}")
        self.transcript.append({"role": "user", "content": user_text})

        if self._transcript_callback:
            self._transcript_callback("user", user_text)

        # 2. Generate response
        logger.info("Generating response...")
        response_text = self.llm.generate_response(user_text)

        if not response_text:
            self._set_state(ConversationState.LISTENING)
            return None

        logger.info(f"Assistant: {response_text}")
        self.transcript.append({"role": "assistant", "content": response_text})

        if self._transcript_callback:
            self._transcript_callback("assistant", response_text)

        # 3. Synthesize speech
        # Strip any *asterisk actions* and [TRANSFER] markers before TTS
        clean_text = re.sub(r'\*[^*]+\*', '', response_text).strip()
        clean_text = re.sub(r'\[TRANSFER\]', '', clean_text).strip()
        clean_text = ' '.join(clean_text.split())  # Collapse multiple spaces

        if not clean_text:
            self._set_state(ConversationState.LISTENING)
            return None

        logger.info("Synthesizing speech...")
        audio_response = self.tts.synthesize(clean_text)

        if not audio_response:
            self._set_state(ConversationState.LISTENING)
            return None

        # Check if we should transfer the call
        if self.llm.should_transfer(response_text):
            self._transfer_number = self.llm.get_transfer_number()
            self._set_state(ConversationState.TRANSFERRING)
        # Check if we should end the call
        elif self.llm.should_end_call(response_text):
            self._set_state(ConversationState.COMPLETED)
        else:
            self._set_state(ConversationState.SPEAKING)

        return audio_response

    def set_speaking(self, is_speaking: bool):
        """
        Set speaking state (for echo suppression).
        Call this when starting/stopping TTS playback.
        """
        self._is_speaking = is_speaking

        if not is_speaking and self.state == ConversationState.SPEAKING:
            # Done speaking, go back to listening
            self._set_state(ConversationState.LISTENING)

    def get_result(self) -> ConversationResult:
        """Get conversation result"""
        summary = ""
        if self.transcript:
            # Use last assistant message as summary
            for msg in reversed(self.transcript):
                if msg["role"] == "assistant":
                    summary = msg["content"]
                    break

        # Consider call successful if:
        # 1. State is COMPLETED (AI said goodbye), OR
        # 2. State is TRANSFERRING (call being transferred), OR
        # 3. There was meaningful conversation (>2 exchanges = 4 messages)
        has_conversation = len(self.transcript) >= 4
        success = (self.state in [ConversationState.COMPLETED, ConversationState.TRANSFERRING]) or has_conversation

        return ConversationResult(
            success=success,
            summary=summary,
            transcript=self.transcript,
            duration_seconds=time.time() - self.start_time,
            transfer_to=self._transfer_number
        )


# Test function
async def test_conversation():
    import sounddevice as sd

    print("Testing local conversation engine...")

    engine = LocalConversationEngine()
    engine.initialize()

    config = ConversationConfig(
        objective="You are calling from City Library to remind someone their book is ready.",
        context={"library": "City Library"}
    )

    engine.start(config)

    def state_callback(state):
        print(f"State: {state.value}")

    def transcript_callback(role, text):
        print(f"{role}: {text}")

    engine.on_state_change(state_callback)
    engine.on_transcript(transcript_callback)

    print("Speak into your microphone...")

    # Audio callback
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(f"Status: {status}")

        audio = indata.flatten()

        # Process audio
        response = engine.process_audio(audio)

        if response:
            # Play response
            engine.set_speaking(True)
            response_array = np.frombuffer(response, dtype=np.int16)

            # Resample to output rate
            output_samples = int(len(response_array) * 48000 / 24000)
            response_48k = np.interp(
                np.linspace(0, len(response_array) - 1, output_samples),
                np.arange(len(response_array)),
                response_array.astype(np.float32)
            ).astype(np.int16)

            sd.play(response_48k, 48000)
            sd.wait()
            engine.set_speaking(False)

        outdata.fill(0)

    # Run for 60 seconds
    with sd.Stream(
        samplerate=48000,
        blocksize=1024,
        channels=1,
        dtype=np.int16,
        callback=audio_callback
    ):
        print("Listening for 60 seconds...")
        await asyncio.sleep(60)

    result = engine.get_result()
    print(f"\nResult: success={result.success}")
    print(f"Transcript: {result.transcript}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_conversation())
