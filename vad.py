"""
Voice Activity Detection using WebRTC VAD

Detects when someone is speaking in audio stream.
Simple, reliable, and doesn't require torch.
"""

import webrtcvad
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """WebRTC VAD wrapper for detecting speech in audio"""

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        max_speech_ms: int = 15000,  # Force end after 15 seconds
        energy_threshold: int = 500,  # RMS threshold for speech detection
    ):
        """
        Initialize VAD.

        Args:
            aggressiveness: 0-3, higher = more aggressive filtering
            sample_rate: Must be 8000, 16000, 32000, or 48000
            frame_duration_ms: Must be 10, 20, or 30
            min_speech_ms: Minimum speech duration to trigger
            min_silence_ms: Silence duration to end speech segment
            max_speech_ms: Maximum speech duration before forcing end
            energy_threshold: Minimum RMS level to consider as speech
        """
        # WebRTC VAD only supports specific sample rates
        if sample_rate not in [8000, 16000, 32000, 48000]:
            # We'll resample to 16kHz
            self.native_sample_rate = sample_rate
            self.vad_sample_rate = 16000
        else:
            self.native_sample_rate = sample_rate
            self.vad_sample_rate = sample_rate

        self.frame_duration_ms = frame_duration_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_speech_ms = max_speech_ms
        self.energy_threshold = energy_threshold

        # Frame size in samples
        self.frame_size = int(self.vad_sample_rate * frame_duration_ms / 1000)

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)

        # State tracking
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_speech_frames = 0  # Track total speech duration

        # Frames needed to trigger
        self._speech_frames_needed = int(min_speech_ms / frame_duration_ms)
        self._silence_frames_needed = int(min_silence_ms / frame_duration_ms)
        self._max_speech_frames = int(max_speech_ms / frame_duration_ms)

        # Audio buffer for accumulating speech
        self._audio_buffer = deque(maxlen=self.native_sample_rate * 30)  # 30 sec max

        # Frame buffer for incomplete frames
        self._frame_buffer = np.array([], dtype=np.int16)

        logger.info(f"VAD initialized (aggressiveness={aggressiveness}, sr={sample_rate})")

    def reset(self):
        """Reset VAD state for new call"""
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_speech_frames = 0
        self._audio_buffer.clear()
        self._frame_buffer = np.array([], dtype=np.int16)

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if from_rate == to_rate:
            return audio

        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float32)).astype(np.int16)

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process an audio chunk and detect speech.

        Args:
            audio_chunk: Audio samples as int16 numpy array

        Returns:
            dict with keys:
                - is_speech: True if speech detected in this chunk
                - speech_started: True if speech just started
                - speech_ended: True if speech just ended
                - audio_buffer: Accumulated audio if speech ended
        """
        result = {
            'is_speech': False,
            'speech_started': False,
            'speech_ended': False,
            'audio_buffer': None
        }

        # Add original audio to buffer (for later transcription)
        self._audio_buffer.extend(audio_chunk)

        # Resample if needed for VAD
        if self.native_sample_rate != self.vad_sample_rate:
            audio_for_vad = self._resample(audio_chunk, self.native_sample_rate, self.vad_sample_rate)
        else:
            audio_for_vad = audio_chunk

        # Add to frame buffer
        self._frame_buffer = np.concatenate([self._frame_buffer, audio_for_vad])

        # Process complete frames
        while len(self._frame_buffer) >= self.frame_size:
            frame = self._frame_buffer[:self.frame_size]
            self._frame_buffer = self._frame_buffer[self.frame_size:]

            # Check if frame contains speech
            try:
                # First check energy level
                rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                has_energy = rms >= self.energy_threshold

                # Only use VAD if energy threshold is met
                if has_energy:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.vad_sample_rate)
                else:
                    is_speech = False
            except Exception as e:
                logger.warning(f"VAD error: {e}")
                is_speech = False

            if is_speech:
                result['is_speech'] = True
                self._speech_frames += 1
                self._silence_frames = 0

                if not self._is_speaking and self._speech_frames >= self._speech_frames_needed:
                    self._is_speaking = True
                    self._total_speech_frames = 0  # Start counting
                    result['speech_started'] = True
                    logger.debug("Speech started")
                elif self._is_speaking:
                    self._total_speech_frames += 1
                    # Check for max speech duration
                    if self._total_speech_frames >= self._max_speech_frames:
                        logger.info(f"Max speech duration reached ({self.max_speech_ms}ms), forcing end")
                        self._is_speaking = False
                        result['speech_ended'] = True
                        result['audio_buffer'] = np.array(list(self._audio_buffer), dtype=np.int16)
                        self._audio_buffer.clear()
                        self._total_speech_frames = 0
            else:
                self._speech_frames = 0

                if self._is_speaking:
                    self._silence_frames += 1
                    self._total_speech_frames += 1  # Still count total time

                    # Check for max speech duration
                    if self._total_speech_frames >= self._max_speech_frames:
                        logger.info(f"Max speech duration reached ({self.max_speech_ms}ms), forcing end")
                        self._is_speaking = False
                        result['speech_ended'] = True
                        result['audio_buffer'] = np.array(list(self._audio_buffer), dtype=np.int16)
                        self._audio_buffer.clear()
                        self._total_speech_frames = 0
                        self._silence_frames = 0
                    elif self._silence_frames >= self._silence_frames_needed:
                        # Speech ended normally
                        self._is_speaking = False
                        result['speech_ended'] = True
                        result['audio_buffer'] = np.array(list(self._audio_buffer), dtype=np.int16)
                        logger.debug(f"Speech ended ({len(result['audio_buffer'])} samples)")

                        # Reset for next utterance
                        self._audio_buffer.clear()
                        self._silence_frames = 0
                        self._total_speech_frames = 0

        return result

    @property
    def is_speaking(self) -> bool:
        """Return True if currently detecting speech"""
        return self._is_speaking


# Test function
def test_vad():
    import sounddevice as sd

    print("Testing VAD - speak into your microphone...")

    vad = VoiceActivityDetector(aggressiveness=2, sample_rate=16000)

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")

        audio = (indata.flatten() * 32767).astype(np.int16)
        result = vad.process_chunk(audio)

        if result['speech_started']:
            print(">>> Speech STARTED")
        elif result['speech_ended']:
            print(f"<<< Speech ENDED ({len(result['audio_buffer'])} samples)")
        elif result['is_speech']:
            print("... speaking")

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.float32,
        blocksize=480,  # 30ms at 16kHz
        callback=audio_callback
    ):
        print("Listening for 30 seconds...")
        sd.sleep(30000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_vad()
