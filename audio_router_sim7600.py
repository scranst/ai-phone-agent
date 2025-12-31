"""
Audio Router for SIM7600 HAT - Manages audio via 3.5mm jack

The SIM7600G-H HAT (B) has a single TRRS 3.5mm combo jack.
Audio flow:
- HAT speaker output → Mac mic input (captures call audio)
- Mac speaker output → HAT mic input (sends AI voice to call)

Requires:
- TRRS splitter cable on HAT side to separate mic/speaker
- USB sound card with separate mic-in and headphone-out
  OR MacBook combo jack with appropriate adapter
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


class AudioRouterSIM7600:
    """Audio router for SIM7600 HAT with 3.5mm audio"""

    # Standard sample rates
    OPENAI_SAMPLE_RATE = 24000  # OpenAI Realtime API
    USB_SAMPLE_RATE = 48000  # Most USB sound cards

    def __init__(self,
                 input_device_name: Optional[str] = None,
                 output_device_name: Optional[str] = None,
                 input_sample_rate: int = 48000,
                 output_sample_rate: int = 48000):
        """
        Initialize audio router.

        Args:
            input_device_name: Name of device receiving HAT speaker output
            output_device_name: Name of device sending to HAT mic input
            input_sample_rate: Sample rate of input device (usually 48000)
            output_sample_rate: Sample rate of output device (usually 48000)
        """
        self.input_device_name = input_device_name
        self.output_device_name = output_device_name
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self._is_running = False
        self._input_queue: queue.Queue = queue.Queue()
        self._recording_buffer: list[bytes] = []
        self._is_recording = False

        # For mixed recording (both sides of conversation)
        self._output_buffer: list[bytes] = []
        self._record_both_sides = True

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

    def get_device_sample_rate(self, device_idx: int) -> float:
        """Get the default sample rate of a device"""
        devices = sd.query_devices()
        if 0 <= device_idx < len(devices):
            return devices[device_idx]["default_samplerate"]
        return 48000  # fallback

    def start(self,
              input_device: Optional[str] = None,
              output_device: Optional[str] = None) -> bool:
        """
        Start audio routing for SIM7600 HAT.

        For SIM7600 calls:
        - Input: USB sound card mic (receives audio from HAT speaker)
        - Output: USB sound card headphone (sends AI voice to HAT mic)
        """
        input_name = input_device or self.input_device_name or config.SIM7600_AUDIO_INPUT
        output_name = output_device or self.output_device_name or config.SIM7600_AUDIO_OUTPUT

        # Find devices
        input_idx = self.find_device(input_name, for_input=True)
        output_idx = self.find_device(output_name, for_input=False)

        if input_idx is None:
            logger.error(f"Input device not found: {input_name}")
            logger.info("Available input devices:")
            for d in self.list_devices():
                if d.max_input_channels > 0:
                    logger.info(f"  [{d.index}] {d.name}")
            return False

        if output_idx is None:
            logger.error(f"Output device not found: {output_name}")
            logger.info("Available output devices:")
            for d in self.list_devices():
                if d.max_output_channels > 0:
                    logger.info(f"  [{d.index}] {d.name}")
            return False

        # Get actual sample rates
        self.input_sample_rate = int(self.get_device_sample_rate(input_idx))
        self.output_sample_rate = int(self.get_device_sample_rate(output_idx))

        logger.info(f"Using input device [{input_idx}]: {input_name} @ {self.input_sample_rate}Hz")
        logger.info(f"Using output device [{output_idx}]: {output_name} @ {self.output_sample_rate}Hz")

        try:
            # Open input stream (capture call audio from HAT)
            self.input_stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                channels=config.CHANNELS,
                dtype=np.int16,
                device=input_idx,
                blocksize=config.CHUNK_SIZE * (self.input_sample_rate // config.SAMPLE_RATE),
                callback=self._input_callback
            )

            # Open output stream (send AI voice to HAT)
            self.output_stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                channels=config.CHANNELS,
                dtype=np.int16,
                device=output_idx,
                blocksize=config.CHUNK_SIZE * (self.output_sample_rate // config.SAMPLE_RATE)
            )

            self.input_stream.start()
            self.output_stream.start()
            self._is_running = True
            logger.info("Audio routing started for SIM7600")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio: {e}")
            self.stop()
            return False

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio between sample rates with proper anti-aliasing"""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate

        if ratio == 0.5:
            # 2x downsample (48kHz -> 24kHz) - MUST anti-alias first!
            # Apply simple moving average low-pass filter before decimation
            # This prevents high frequencies from aliasing into audible range
            audio_f = audio.astype(np.float32)

            # Simple 5-tap low-pass FIR filter (removes >12kHz content)
            # Coefficients designed for -3dB at ~10kHz, strong roll-off above
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)

            # Apply filter using convolution
            filtered = np.convolve(audio_f, kernel, mode='same')

            # Then decimate (take every other sample)
            decimated = filtered[0::2]

            return np.clip(decimated, -32768, 32767).astype(np.int16)

        elif ratio == 2.0:
            # 2x upsample (24kHz -> 48kHz) with linear interpolation
            audio_f = audio.astype(np.float32)
            resampled = np.zeros(len(audio) * 2, dtype=np.float32)
            resampled[0::2] = audio_f
            # Linear interpolation between samples
            resampled[1::2] = (audio_f + np.roll(audio_f, -1)) / 2
            resampled[-1] = audio_f[-1]  # Fix last sample
            return resampled.astype(np.int16)
        else:
            # General resampling using linear interpolation
            old_indices = np.arange(len(audio))
            new_length = int(len(audio) * ratio)
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(new_indices, old_indices, audio.astype(np.float32)).astype(audio.dtype)

    def _input_callback(self, indata, frames, time, status):
        """Callback for input stream - receives audio from SIM7600"""
        if status:
            logger.warning(f"Input stream status: {status}")

        # Flatten and resample to OpenAI's 24kHz
        audio = indata.flatten()
        audio_24k = self._resample(audio, self.input_sample_rate, self.OPENAI_SAMPLE_RATE)

        # Check audio levels - log periodically
        rms = np.sqrt(np.mean(audio_24k.astype(np.float32)**2))
        peak = np.max(np.abs(audio_24k))

        # Log audio stats every ~1 second (about 47 chunks at 24kHz/512 samples)
        if not hasattr(self, '_chunk_count'):
            self._chunk_count = 0
        self._chunk_count += 1
        if self._chunk_count % 50 == 0:
            logger.info(f"Audio levels - RMS: {rms:.0f}, Peak: {peak}")

        audio_bytes = audio_24k.tobytes()

        # Add to queue for async processing
        self._input_queue.put(audio_bytes)

        # Add to recording buffer if recording
        if self._is_recording:
            self._recording_buffer.append(audio_bytes)

    async def read_audio(self) -> Optional[bytes]:
        """Read audio chunk from input (call audio from SIM7600)"""
        try:
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

    def write_audio(self, audio_data: bytes):
        """Write audio to output (send AI voice to SIM7600)"""
        if self.output_stream and self._is_running:
            try:
                # Convert bytes to numpy array (24kHz from OpenAI)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Resample to output device rate
                audio_array = self._resample(audio_array, self.OPENAI_SAMPLE_RATE, self.output_sample_rate)

                # Reshape to (frames, channels)
                audio_array = audio_array.reshape(-1, config.CHANNELS)
                self.output_stream.write(audio_array)

                # Also record AI output if recording both sides
                if self._is_recording and self._record_both_sides:
                    self._output_buffer.append(audio_data)

            except Exception as e:
                logger.error(f"Error writing audio: {e}")

    async def write_audio_async(self, audio_data: bytes):
        """Async wrapper for writing audio"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.write_audio, audio_data)

    def start_recording(self, record_both_sides: bool = True):
        """Start recording the call"""
        self._recording_buffer = []
        self._output_buffer = []
        self._is_recording = True
        self._record_both_sides = record_both_sides
        logger.info("Started recording")

    def stop_recording(self, save_path: Optional[str] = None) -> Optional[str]:
        """Stop recording and save to file"""
        self._is_recording = False

        if not self._recording_buffer:
            return None

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(config.CALLS_DIR, f"call_{timestamp}.wav")

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # Mix input and output if recording both sides
            if self._record_both_sides and self._output_buffer:
                mixed_audio = self._mix_audio(self._recording_buffer, self._output_buffer)
                audio_data = mixed_audio
            else:
                audio_data = b"".join(self._recording_buffer)

            with wave.open(save_path, "wb") as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.OPENAI_SAMPLE_RATE)  # Save at 24kHz
                wf.writeframes(audio_data)

            logger.info(f"Recording saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return None

    def _mix_audio(self, input_chunks: list[bytes], output_chunks: list[bytes]) -> bytes:
        """Mix input and output audio into single track"""
        # Convert to arrays
        input_audio = np.frombuffer(b"".join(input_chunks), dtype=np.int16)
        output_audio = np.frombuffer(b"".join(output_chunks), dtype=np.int16)

        # Make same length
        max_len = max(len(input_audio), len(output_audio))
        input_padded = np.zeros(max_len, dtype=np.int16)
        output_padded = np.zeros(max_len, dtype=np.int16)
        input_padded[:len(input_audio)] = input_audio
        output_padded[:len(output_audio)] = output_audio

        # Mix (with clipping protection)
        mixed = np.clip(
            input_padded.astype(np.int32) + output_padded.astype(np.int32),
            -32768, 32767
        ).astype(np.int16)

        return mixed.tobytes()

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
    print("SIM7600 Audio Router - Device Test")
    print("=" * 60)

    router = AudioRouterSIM7600()

    print("\nAvailable audio devices:")
    print("-" * 60)
    for device in router.list_devices():
        direction = []
        if device.max_input_channels > 0:
            direction.append("IN")
        if device.max_output_channels > 0:
            direction.append("OUT")
        rate = f"{int(device.default_sample_rate)}Hz"
        print(f"[{device.index:2}] {device.name:40} {'/'.join(direction):8} {rate}")
    print("-" * 60)

    # Look for USB sound card
    usb_devices = [d for d in router.list_devices() if "usb" in d.name.lower()]
    if usb_devices:
        print("\nDetected USB audio devices:")
        for d in usb_devices:
            print(f"  - {d.name}")
    else:
        print("\nNo USB audio device detected.")
        print("Connect a USB sound card for SIM7600 HAT audio.")

    print("\nExpected setup for SIM7600 HAT (B):")
    print("  1. TRRS splitter on HAT's 3.5mm jack")
    print("  2. HAT speaker (green) → USB sound card mic input")
    print("  3. USB sound card headphone output → HAT mic (pink)")


if __name__ == "__main__":
    test_audio_router()
