"""
Ringback Tone Detector using Goertzel Algorithm

Detects US ringback tones (440Hz + 480Hz dual-tone) to determine
when someone answers vs when the phone is still ringing.

US Ringback: 440Hz + 480Hz, 2 seconds on, 4 seconds off
"""

import numpy as np
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class RingbackDetector:
    """Detects ringback tones using Goertzel algorithm"""

    # US ringback tone frequencies
    FREQ_1 = 440  # Hz
    FREQ_2 = 480  # Hz

    # Detection thresholds (tune these based on testing)
    TONE_THRESHOLD = 500  # Minimum magnitude to consider tone present
    TONE_RATIO_THRESHOLD = 0.3  # Ratio of tone energy to total energy

    # Cadence tracking (US: 2s on, 4s off)
    RINGBACK_ON_MS = 2000
    RINGBACK_OFF_MS = 4000

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

        # History for cadence detection
        self.tone_history = deque(maxlen=100)  # Last 100 samples (~2-3 seconds)
        self.ringback_count = 0
        self.last_ringback_time = 0
        self._answered = False

    def goertzel(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm - efficiently detect magnitude of single frequency.

        More efficient than FFT when only detecting a few frequencies.
        """
        N = len(samples)
        if N == 0:
            return 0.0

        # Normalize to prevent overflow
        samples = samples.astype(np.float64) / 32768.0

        k = int(0.5 + N * target_freq / self.sample_rate)
        omega = 2 * np.pi * k / N
        coeff = 2 * np.cos(omega)

        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0

        # Calculate magnitude
        magnitude = np.sqrt(s1 * s1 + s2 * s2 - coeff * s1 * s2)
        return magnitude

    def is_ringback(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains ringback tone.

        Args:
            audio_chunk: numpy array of int16 audio samples

        Returns:
            True if ringback tone detected, False otherwise
        """
        if len(audio_chunk) < 100:
            return False

        # Detect both ringback frequencies
        mag_440 = self.goertzel(audio_chunk, self.FREQ_1)
        mag_480 = self.goertzel(audio_chunk, self.FREQ_2)

        # Calculate total energy for comparison
        total_energy = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)) / 32768.0

        # Check if both tones are present and strong relative to total
        tone_present = (
            mag_440 > self.TONE_THRESHOLD and
            mag_480 > self.TONE_THRESHOLD
        )

        # Log for debugging
        if mag_440 > 100 or mag_480 > 100:
            logger.debug(f"Tone magnitudes: 440Hz={mag_440:.0f}, 480Hz={mag_480:.0f}, total={total_energy:.4f}")

        return tone_present

    def process_audio(self, audio_bytes: bytes) -> dict:
        """
        Process audio chunk and return detection status.

        Args:
            audio_bytes: Raw audio bytes (int16, mono)

        Returns:
            dict with:
                - is_ringback: True if ringback detected
                - is_voice: True if voice (non-ringback) detected
                - answered: True if call appears answered (ringback stopped)
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Check for ringback
        ringback_now = self.is_ringback(audio)

        # Track history
        self.tone_history.append(ringback_now)

        # Check for voice (audio present but not ringback)
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        has_audio = rms > 200  # Above noise floor
        is_voice = has_audio and not ringback_now

        # Detect answer: had ringback, now have voice
        if not self._answered:
            # Check if we had ringback before
            ringback_history = list(self.tone_history)
            had_ringback = sum(ringback_history[:-10]) > 5 if len(ringback_history) > 10 else False
            recent_no_ringback = sum(ringback_history[-10:]) < 2 if len(ringback_history) >= 10 else False

            if had_ringback and recent_no_ringback and is_voice:
                logger.info("Call answered detected (ringback stopped, voice started)")
                self._answered = True

        return {
            "is_ringback": ringback_now,
            "is_voice": is_voice,
            "answered": self._answered,
            "rms": rms
        }

    def reset(self):
        """Reset detector state for new call"""
        self.tone_history.clear()
        self.ringback_count = 0
        self._answered = False


def test_detector():
    """Test the ringback detector with synthetic tones"""
    import time

    detector = RingbackDetector(sample_rate=24000)

    # Generate synthetic ringback tone (440 + 480 Hz)
    duration = 0.1  # 100ms chunk
    t = np.linspace(0, duration, int(24000 * duration))
    ringback = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 480 * t)
    )
    ringback_int16 = (ringback * 16000).astype(np.int16)

    # Generate synthetic voice (complex waveform)
    voice = np.random.randn(len(t)) * 1000
    voice_int16 = voice.astype(np.int16)

    # Generate silence
    silence = np.zeros(len(t), dtype=np.int16)

    print("Testing ringback detection:")
    print("-" * 40)

    # Test ringback
    result = detector.process_audio(ringback_int16.tobytes())
    print(f"Ringback tone: is_ringback={result['is_ringback']}, is_voice={result['is_voice']}")

    # Test voice
    result = detector.process_audio(voice_int16.tobytes())
    print(f"Voice: is_ringback={result['is_ringback']}, is_voice={result['is_voice']}")

    # Test silence
    result = detector.process_audio(silence.tobytes())
    print(f"Silence: is_ringback={result['is_ringback']}, is_voice={result['is_voice']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_detector()
