"""
Speech-to-Text using faster-whisper

Transcribes audio to text using OpenAI's Whisper model running locally.
"""

import numpy as np
import logging
from typing import Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class SpeechToText:
    """faster-whisper wrapper for local speech recognition"""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize Whisper model.

        Args:
            model_size: Model size (tiny.en, base.en, small.en, medium.en, large-v3)
            device: Device to run on (cpu, cuda, auto)
            compute_type: Computation type (int8, float16, float32)
        """
        logger.info(f"Loading Whisper model: {model_size}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        self.sample_rate = 16000  # Whisper expects 16kHz
        logger.info(f"Whisper model loaded: {model_size}")

        # Warm up the model with a tiny silent audio
        self._warmup()

    def _warmup(self):
        """Warm up the model to avoid cold start delay on first transcription"""
        logger.info("Warming up Whisper model...")
        silent = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        list(self.model.transcribe(silent, language="en"))
        logger.info("Whisper warmup complete")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en"
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as int16 numpy array
            sample_rate: Sample rate of audio
            language: Language code

        Returns:
            Transcribed text
        """
        if len(audio) == 0:
            return ""

        # Convert to float32 normalized to [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            duration = len(audio_float) / sample_rate
            new_length = int(duration * 16000)
            indices = np.linspace(0, len(audio_float) - 1, new_length)
            audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)

        # Ensure float32 (Whisper requirement)
        audio_float = np.ascontiguousarray(audio_float, dtype=np.float32)

        # Transcribe
        try:
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=5,
                language=language,
                vad_filter=False  # Disabled - we have our own VAD
            )

            # Combine all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            text = " ".join(text_parts).strip()

            if text:
                logger.info(f"Transcribed: {text}")

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str = "en"
    ) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio bytes (int16 PCM)
            sample_rate: Sample rate of audio
            language: Language code

        Returns:
            Transcribed text
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return self.transcribe(audio, sample_rate, language)


# Test function
def test_stt():
    import sounddevice as sd
    import time

    print("Testing STT - recording 5 seconds of audio...")

    stt = SpeechToText(model_size="base.en")

    # Record audio
    duration = 5
    sample_rate = 16000
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()

    print("Transcribing...")
    start = time.time()
    text = stt.transcribe(audio.flatten(), sample_rate)
    elapsed = time.time() - start

    print(f"Transcription ({elapsed:.2f}s): {text}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_stt()
