"""
Text-to-Speech using Piper TTS

Converts text to natural-sounding speech using local neural TTS.
"""

import numpy as np
import logging
import wave
import io
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models", "piper", "en_US-lessac-medium.onnx"
)


class TextToSpeech:
    """Piper TTS wrapper for local speech synthesis"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        output_sample_rate: int = 24000
    ):
        """
        Initialize Piper TTS.

        Args:
            model_path: Path to .onnx voice model
            output_sample_rate: Desired output sample rate
        """
        from piper import PiperVoice

        logger.info(f"Loading Piper voice: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model not found: {model_path}")

        self.voice = PiperVoice.load(model_path)
        self.output_sample_rate = output_sample_rate

        # Piper outputs at 22050Hz by default
        self.native_sample_rate = 22050

        logger.info(f"Piper TTS loaded (native={self.native_sample_rate}Hz, output={output_sample_rate}Hz)")

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            Raw audio bytes (int16 PCM at output_sample_rate)
        """
        if not text or not text.strip():
            return b""

        try:
            # Synthesize using generator
            audio_bytes_list = []
            for chunk in self.voice.synthesize(text):
                audio_bytes_list.append(chunk.audio_int16_bytes)
                # Update native sample rate from first chunk
                self.native_sample_rate = chunk.sample_rate

            if not audio_bytes_list:
                logger.warning("No audio generated")
                return b""

            all_audio = b''.join(audio_bytes_list)
            audio = np.frombuffer(all_audio, dtype=np.int16)

            # Resample if needed
            if self.output_sample_rate != self.native_sample_rate:
                audio = self._resample(audio, self.native_sample_rate, self.output_sample_rate)

            logger.debug(f"Synthesized: '{text[:50]}...' -> {len(audio)} samples")

            return audio.tobytes()

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

    def synthesize_to_array(self, text: str) -> np.ndarray:
        """
        Convert text to numpy array.

        Args:
            text: Text to synthesize

        Returns:
            Audio as int16 numpy array
        """
        audio_bytes = self.synthesize(text)
        if audio_bytes:
            return np.frombuffer(audio_bytes, dtype=np.int16)
        return np.array([], dtype=np.int16)

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if from_rate == to_rate:
            return audio

        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)

        return np.interp(
            indices,
            np.arange(len(audio)),
            audio.astype(np.float32)
        ).astype(np.int16)

    def get_audio_duration(self, audio_bytes: bytes) -> float:
        """Get duration of audio in seconds"""
        num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
        return num_samples / self.output_sample_rate


# Test function
def test_tts():
    import sounddevice as sd

    print("Testing TTS...")

    tts = TextToSpeech(output_sample_rate=24000)

    text = "Hello! This is a test of the Piper text to speech system. How does it sound?"
    print(f"Synthesizing: {text}")

    audio = tts.synthesize_to_array(text)
    duration = len(audio) / 24000

    print(f"Generated {len(audio)} samples ({duration:.2f}s)")
    print("Playing...")

    # Resample to 48kHz for playback (common output rate)
    audio_48k = np.interp(
        np.linspace(0, len(audio) - 1, len(audio) * 2),
        np.arange(len(audio)),
        audio.astype(np.float32)
    ).astype(np.int16)

    sd.play(audio_48k, 48000)
    sd.wait()

    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_tts()
