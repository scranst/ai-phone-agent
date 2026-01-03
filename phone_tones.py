"""
Phone Tone Detection

Detects standard US phone tones by frequency analysis.
Used to filter out non-speech audio and detect call state changes.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Standard US phone tone frequencies (Hz)
PHONE_TONES = {
    'dial_tone': (350, 440),           # Continuous
    'ringback': (440, 480),            # 2s on, 4s off
    'busy': (480, 620),                # 0.5s on, 0.5s off
    'reorder': (480, 620),             # 0.25s on, 0.25s off (fast busy)
    'off_hook': (1400, 2060, 2450, 2600),  # Loud warning
}

# DTMF frequencies
DTMF_LOW = (697, 770, 852, 941)
DTMF_HIGH = (1209, 1336, 1477, 1633)


class PhoneToneDetector:
    """Detects standard phone tones using FFT frequency analysis"""

    def __init__(self, sample_rate: int = 24000, tolerance_hz: int = 20):
        """
        Args:
            sample_rate: Audio sample rate
            tolerance_hz: Frequency matching tolerance (+/- Hz)
        """
        self.sample_rate = sample_rate
        self.tolerance = tolerance_hz

        # Build list of all phone tone frequencies to detect
        self.tone_frequencies = set()
        for freqs in PHONE_TONES.values():
            self.tone_frequencies.update(freqs)
        self.tone_frequencies.update(DTMF_LOW)
        self.tone_frequencies.update(DTMF_HIGH)

        # State tracking
        self._was_ringback = False
        self._ringback_count = 0
        self._ringback_ended_fired = False

        logger.info(f"PhoneToneDetector initialized (sr={sample_rate}, tol={tolerance_hz}Hz)")

    def detect_tones(self, audio: np.ndarray) -> dict:
        """
        Analyze audio for phone tones.

        Args:
            audio: Audio samples as int16 numpy array

        Returns:
            dict with:
                - is_phone_tone: True if audio is a phone tone
                - tone_type: Name of detected tone (or None)
                - is_ringback: True if ringback tone detected
                - ringback_ended: True if ringback just stopped (call answered)
                - dominant_freqs: List of dominant frequencies found
        """
        result = {
            'is_phone_tone': False,
            'tone_type': None,
            'is_ringback': False,
            'ringback_ended': False,
            'dominant_freqs': []
        }

        # Need enough samples for frequency resolution
        if len(audio) < 512:
            return result

        # Convert to float and normalize
        audio_float = audio.astype(np.float32) / 32768.0

        # Apply window to reduce spectral leakage
        window = np.hanning(len(audio_float))
        windowed = audio_float * window

        # FFT
        fft = np.fft.rfft(windowed)
        magnitudes = np.abs(fft)
        freqs = np.fft.rfftfreq(len(windowed), 1.0 / self.sample_rate)

        # Find dominant frequencies using local maxima (true peaks)
        threshold = np.max(magnitudes) * 0.5  # 50% of max

        dominant_freqs = []
        for i in range(1, len(magnitudes) - 1):
            freq = freqs[i]
            # Only consider frequencies in phone tone range (300-3000 Hz)
            if 300 <= freq <= 3000:
                # Check if this is a local maximum above threshold
                if (magnitudes[i] > threshold and
                    magnitudes[i] > magnitudes[i-1] and
                    magnitudes[i] > magnitudes[i+1]):
                    dominant_freqs.append(freq)

        if not dominant_freqs:
            return result

        result['dominant_freqs'] = dominant_freqs

        # Phone tones have very few distinct frequencies (2-4)
        # Speech has many frequencies spread across the spectrum
        # If we detect more than 6 peaks, it's probably speech, not a tone
        if len(dominant_freqs) > 6:
            return result

        # Check if frequencies match known phone tones
        for tone_name, tone_freqs in PHONE_TONES.items():
            if self._matches_tone(dominant_freqs, tone_freqs):
                result['is_phone_tone'] = True
                result['tone_type'] = tone_name

                if tone_name == 'ringback':
                    result['is_ringback'] = True
                    self._ringback_count += 1
                break

        # Check for DTMF (must have exactly 2 strong frequencies)
        if not result['is_phone_tone'] and len(dominant_freqs) <= 3:
            if self._is_dtmf(dominant_freqs):
                result['is_phone_tone'] = True
                result['tone_type'] = 'dtmf'

        # Detect when ringback ends (call answered) - only fire once
        if not self._ringback_ended_fired:
            if self._was_ringback and not result['is_ringback']:
                # Ringback stopped - likely call was answered
                if self._ringback_count >= 2:  # Had at least 2 ringback detections
                    result['ringback_ended'] = True
                    self._ringback_ended_fired = True
                    logger.info("Ringback ended - call likely answered")

        self._was_ringback = result['is_ringback']

        return result

    def _cluster_frequencies(self, freqs: list, cluster_range: int = 25) -> list:
        """Cluster nearby frequencies together (smaller range to keep tones separate)"""
        if not freqs:
            return []

        freqs = sorted(freqs)
        clusters = []
        current_cluster = [freqs[0]]

        for freq in freqs[1:]:
            if freq - current_cluster[-1] <= cluster_range:
                current_cluster.append(freq)
            else:
                # Average the cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [freq]

        clusters.append(np.mean(current_cluster))
        return clusters

    def _matches_tone(self, detected_freqs: list, tone_freqs: tuple) -> bool:
        """Check if detected frequencies match a known tone"""
        if len(detected_freqs) < len(tone_freqs):
            return False

        matches = 0
        for tone_freq in tone_freqs:
            for detected in detected_freqs:
                if abs(detected - tone_freq) <= self.tolerance:
                    matches += 1
                    break

        # Require all tone frequencies to be present
        return matches == len(tone_freqs)

    def _is_dtmf(self, detected_freqs: list) -> bool:
        """Check if frequencies match DTMF pattern (one low + one high)"""
        has_low = False
        has_high = False

        for freq in detected_freqs:
            for dtmf_freq in DTMF_LOW:
                if abs(freq - dtmf_freq) <= self.tolerance:
                    has_low = True
                    break
            for dtmf_freq in DTMF_HIGH:
                if abs(freq - dtmf_freq) <= self.tolerance:
                    has_high = True
                    break

        return has_low and has_high

    def reset(self):
        """Reset state for new call"""
        self._was_ringback = False
        self._ringback_count = 0
        self._ringback_ended_fired = False


def test_tone_detector():
    """Test with synthetic tones"""
    print("Testing phone tone detector...")

    detector = PhoneToneDetector(sample_rate=24000)

    # Generate ringback tone (440 + 480 Hz)
    duration = 0.5  # 500ms
    t = np.linspace(0, duration, int(24000 * duration))
    ringback = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * 16000
    ringback = ringback.astype(np.int16)

    result = detector.detect_tones(ringback)
    print(f"Ringback test: {result}")
    assert result['is_ringback'], "Should detect ringback"

    # Generate speech-like audio (random with some frequency content)
    np.random.seed(42)
    speech = np.random.randn(int(24000 * 0.5)) * 3000
    speech = speech.astype(np.int16)

    result = detector.detect_tones(speech)
    print(f"Speech test: {result}")
    assert not result['is_phone_tone'], "Should not detect speech as phone tone"

    print("All tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tone_detector()
