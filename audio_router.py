"""
Audio Router - Manages audio capture and playback through virtual audio devices

Uses sounddevice to:
- Capture incoming audio from BlackHole (call audio)
- Play outgoing audio to BlackHole (AI voice)
- Handle audio format conversion for OpenAI Realtime API
"""

import sounddevice as sd
import numpy as np
import asyncio
import logging
from typing import Optional, Callable, AsyncIterator
from dataclasses import dataclass
import wave
import os
from datetime import datetime
import threading
import queue

import config

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float


class AudioRouter:
    # BlackHole native sample rate
    BLACKHOLE_SAMPLE_RATE = 48000

    def __init__(self):
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self._is_running = False
        self._input_queue: queue.Queue = queue.Queue()
        self._recording_buffer: list[bytes] = []
        self._is_recording = False

    def list_devices(self) -> list[AudioDevice]:
        """List all available audio devices"""
        devices = []
        device_list = sd.query_devices()
        for i, info in enumerate(device_list):
            devices.append(AudioDevice(
                index=i,
                name=info["name"],
                max_input_channels=info["max_input_channels"],
                max_output_channels=info["max_output_channels"],
                default_sample_rate=info["default_samplerate"]
            ))
        return devices

    def find_device(self, name: str, for_input: bool = True) -> Optional[int]:
        """Find a device index by name"""
        for device in self.list_devices():
            if name.lower() in device.name.lower():
                if for_input and device.max_input_channels > 0:
                    return device.index
                elif not for_input and device.max_output_channels > 0:
                    return device.index
        return None

    def start(self,
              input_device: Optional[str] = None,
              output_device: Optional[str] = None) -> bool:
        """
        Start audio routing

        For phone calls:
        - Input: BlackHole (receives audio from FaceTime call)
        - Output: BlackHole (sends AI voice to FaceTime call)
        """
        input_name = input_device or config.AUDIO_INPUT_DEVICE
        output_name = output_device or config.AUDIO_OUTPUT_DEVICE

        # Find devices
        input_idx = self.find_device(input_name, for_input=True)
        output_idx = self.find_device(output_name, for_input=False)

        if input_idx is None:
            logger.error(f"Input device not found: {input_name}")
            logger.info("Available devices:")
            for d in self.list_devices():
                if d.max_input_channels > 0:
                    logger.info(f"  [{d.index}] {d.name}")
            return False

        if output_idx is None:
            logger.error(f"Output device not found: {output_name}")
            return False

        logger.info(f"Using input device [{input_idx}]: {input_name}")
        logger.info(f"Using output device [{output_idx}]: {output_name}")

        try:
            # Open input stream (capture call audio)
            # Use BlackHole's native 48kHz - we'll resample to 24kHz for OpenAI
            self.input_stream = sd.InputStream(
                samplerate=self.BLACKHOLE_SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.int16,
                device=input_idx,
                blocksize=config.CHUNK_SIZE * 2,  # Larger buffer for 48kHz
                callback=self._input_callback
            )

            # Open output stream to BlackHole - this sends AI voice to FaceTime
            self.output_stream = sd.OutputStream(
                samplerate=self.BLACKHOLE_SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.int16,
                device=output_idx,
                blocksize=config.CHUNK_SIZE * 2
            )
            logger.info(f"Output stream using device [{output_idx}]: {output_name}")

            self.input_stream.start()
            self.output_stream.start()
            self._is_running = True
            logger.info("Audio routing started")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio: {e}")
            self.stop()
            return False

    def _resample_48k_to_24k(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from 48kHz to 24kHz (2x downsample)"""
        # Take every other sample for 2x downsampling
        return audio[0::2]

    def _input_callback(self, indata, frames, time, status):
        """Callback for input stream - receives audio from call"""
        if status:
            logger.warning(f"Input stream status: {status}")

        # Flatten input and resample from 48kHz to 24kHz for OpenAI
        audio_48k = indata.flatten()
        audio_24k = self._resample_48k_to_24k(audio_48k)
        audio_bytes = audio_24k.tobytes()

        # Add to queue for async processing
        self._input_queue.put(audio_bytes)

        # Add to recording buffer if recording (save at 24kHz)
        if self._is_recording:
            self._recording_buffer.append(audio_bytes)

    async def read_audio(self) -> Optional[bytes]:
        """Read audio chunk from input (call audio)"""
        try:
            # Non-blocking read from queue
            return self._input_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            return None

    async def read_audio_stream(self) -> AsyncIterator[bytes]:
        """Async iterator for continuous audio reading"""
        while self._is_running:
            audio = await self.read_audio()
            if audio:
                yield audio

    def _resample_24k_to_48k(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from 24kHz to 48kHz (2x upsample)"""
        # Simple linear interpolation for 2x upsampling
        resampled = np.zeros(len(audio) * 2, dtype=audio.dtype)
        resampled[0::2] = audio  # Original samples at even indices
        resampled[1::2] = audio  # Duplicate at odd indices (simple nearest-neighbor)
        return resampled

    def write_audio(self, audio_data: bytes):
        """Write audio to output (send to call)"""
        if self.output_stream and self._is_running:
            try:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Resample from 24kHz to 48kHz for BlackHole
                audio_array = self._resample_24k_to_48k(audio_array)

                # Reshape to (frames, channels)
                audio_array = audio_array.reshape(-1, config.CHANNELS)
                self.output_stream.write(audio_array)
            except Exception as e:
                logger.error(f"Error writing audio: {e}")

    async def write_audio_async(self, audio_data: bytes):
        """Async wrapper for writing audio"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.write_audio, audio_data)

    def start_recording(self):
        """Start recording the call"""
        self._recording_buffer = []
        self._is_recording = True
        logger.info("Started recording")

    def stop_recording(self, save_path: Optional[str] = None) -> Optional[str]:
        """Stop recording and optionally save to file"""
        self._is_recording = False

        if not self._recording_buffer:
            return None

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(config.CALLS_DIR, f"call_{timestamp}.wav")

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save as WAV
        try:
            with wave.open(save_path, "wb") as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(b"".join(self._recording_buffer))

            logger.info(f"Recording saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return None

    def stop(self):
        """Stop audio routing"""
        self._is_running = False

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None

        logger.info("Audio routing stopped")


def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 for OpenAI API"""
    import base64
    return base64.b64encode(audio_bytes).decode("utf-8")


def base64_to_audio(base64_str: str) -> bytes:
    """Convert base64 from OpenAI API to audio bytes"""
    import base64
    return base64.b64decode(base64_str)


# Test function
def test_audio_router():
    router = AudioRouter()

    print("Available audio devices:")
    print("-" * 60)
    for device in router.list_devices():
        direction = []
        if device.max_input_channels > 0:
            direction.append("IN")
        if device.max_output_channels > 0:
            direction.append("OUT")
        print(f"[{device.index}] {device.name} ({'/'.join(direction)})")
    print("-" * 60)

    blackhole_in = router.find_device("BlackHole", for_input=True)
    blackhole_out = router.find_device("BlackHole", for_input=False)

    if blackhole_in is not None:
        print(f"✓ BlackHole input found at index {blackhole_in}")
    else:
        print("✗ BlackHole input NOT found - install with: brew install blackhole-2ch")

    if blackhole_out is not None:
        print(f"✓ BlackHole output found at index {blackhole_out}")
    else:
        print("✗ BlackHole output NOT found")


if __name__ == "__main__":
    test_audio_router()
