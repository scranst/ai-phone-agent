# AI Phone Agent - Complete Developer Specification

## Overview

Build an AI phone agent that makes outbound calls using a SIM7600G-H 4G HAT connected to a Mac. The agent listens to the person on the phone, understands their speech, generates intelligent responses, and speaks back to them.

**Cost-Optimized Architecture:**
- Local Speech-to-Text (Whisper) - FREE
- Cheap LLM for responses (Claude Haiku or local Ollama) - ~$0.001/call
- Local Text-to-Speech (Piper TTS) - FREE

---

## Hardware Setup

### Components
1. **SIM7600G-H 4G HAT (B)** - Waveshare cellular modem
   - Documentation: https://www.waveshare.com/wiki/SIM7600G-H_4G_HAT_(B)
   - Has a single TRRS 3.5mm combo audio jack
   - Connects to Mac via USB for AT commands

2. **USB Sound Card** - "USB Advanced Audio Device" or similar
   - Must support 48kHz sample rate
   - Separate mic-in and headphone-out jacks

3. **TRRS Splitter Cable** - Splits the HAT's combo jack into separate mic/speaker

4. **Audio Cable Routing:**
   ```
   HAT 3.5mm jack → TRRS Splitter
                    ├── Speaker output (green) → USB sound card MIC INPUT
                    └── Mic input (pink) ← USB sound card HEADPHONE OUTPUT
   ```

### Why This Setup
The SIM7600G-H HAT (B) routes call audio through its 3.5mm jack, NOT through USB audio. The USB connection is only for AT commands. You MUST use an external sound card to capture/play call audio.

---

## Software Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Phone Agent                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Modem      │    │    Audio     │    │ Conversation │  │
│  │  Controller  │    │    Router    │    │    Engine    │  │
│  │  (AT Cmds)   │    │  (48k↔24k)   │    │  (STT+LLM+   │  │
│  │              │    │              │    │     TTS)     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  SIM7600     │    │  USB Sound   │    │ Local Models │  │
│  │  HAT (USB)   │    │    Card      │    │ Whisper/Piper│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Module 1: Modem Controller (sim7600_modem.py)

### USB Connection
```python
import usb.core
import usb.util

# Find the modem
SIMCOM_VENDOR_ID = 0x1e0e
SIM7600_PRODUCT_ID = 0x9001

device = usb.core.find(idVendor=SIMCOM_VENDOR_ID, idProduct=SIM7600_PRODUCT_ID)

# Use interface 2 for AT commands (NOT interface 0 or 1)
INTERFACE_NUM = 2
```

### Critical AT Commands

```python
# Basic setup
"AT"                    # Test connection
"AT+CSQ"                # Check signal strength (should be >10)
"AT+CREG?"              # Check network registration

# Making a call
"ATD{phone_number};"    # Dial (semicolon is required for voice call)

# Audio configuration - MUST SET THESE BEFORE/DURING CALL
"AT+CSDVC=1"            # Route audio to 3.5mm headset jack (CRITICAL!)
"AT+CLVL=2"             # Set volume (1-5, use 2 to avoid clipping)

# Call state monitoring
"AT+CLCC"               # List current calls with state

# Hang up
"AT+CHUP"               # Hang up call
```

### CLCC Response Parsing (CRITICAL BUG FIX)

The CLCC response format is: `+CLCC: id,dir,stat,mode,mpty[,number,type]`

**State codes (stat field - position 2, zero-indexed):**
- 0 = Active (call answered and connected)
- 1 = Held
- 2 = Dialing (MO call)
- 3 = Alerting (ringing at remote end)
- 4 = Incoming (MT call)
- 5 = Waiting

**WRONG (causes false "connected" detection):**
```python
# DON'T DO THIS - matches dir=0, not stat=0
if ",0," in response:
    return "connected"
```

**CORRECT:**
```python
# Parse CLCC properly
# Example: +CLCC: 1,0,0,0,0 means id=1, dir=0(MO), stat=0(Active), mode=0(voice)
match = re.search(r'\+CLCC:\s*(\d+),(\d+),(\d+),(\d+),(\d+)', response)
if match:
    stat = int(match.group(3))  # Third field is state
    if stat == 0:
        return "connected"  # Actually answered
    elif stat == 2:
        return "dialing"
    elif stat == 3:
        return "ringing"
```

### Modem Connection Code

```python
import usb.core
import usb.util
import time

class SIM7600Modem:
    VENDOR_ID = 0x1e0e
    PRODUCT_ID = 0x9001
    INTERFACE = 2

    def __init__(self):
        self.device = None
        self.ep_in = None
        self.ep_out = None

    def connect(self) -> bool:
        self.device = usb.core.find(
            idVendor=self.VENDOR_ID,
            idProduct=self.PRODUCT_ID
        )
        if not self.device:
            return False

        # Detach kernel driver if needed (Linux/Mac)
        if self.device.is_kernel_driver_active(self.INTERFACE):
            self.device.detach_kernel_driver(self.INTERFACE)

        # Set configuration and claim interface
        self.device.set_configuration()
        usb.util.claim_interface(self.device, self.INTERFACE)

        # Find endpoints
        cfg = self.device.get_active_configuration()
        intf = cfg[(self.INTERFACE, 0)]

        self.ep_out = usb.util.find_descriptor(
            intf, custom_match=lambda e:
            usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        )
        self.ep_in = usb.util.find_descriptor(
            intf, custom_match=lambda e:
            usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )

        return self._send_at("AT") is not None

    def _send_at(self, command: str, timeout: float = 1.0) -> str:
        """Send AT command and return response"""
        self.ep_out.write(f"{command}\r\n".encode())

        response = ""
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                data = self.ep_in.read(512, timeout=100)
                response += bytes(data).decode('utf-8', errors='ignore')
                if "OK" in response or "ERROR" in response:
                    break
            except usb.core.USBTimeoutError:
                continue

        return response

    def dial(self, phone_number: str) -> bool:
        """Initiate a call"""
        # Configure audio FIRST
        self._send_at("AT+CSDVC=1")  # Route to headset jack
        self._send_at("AT+CLVL=2")   # Set volume (avoid clipping)

        # Dial
        response = self._send_at(f"ATD{phone_number};", timeout=5.0)
        return "OK" in response

    def get_call_state(self) -> str:
        """Get current call state"""
        response = self._send_at("AT+CLCC")

        match = re.search(r'\+CLCC:\s*(\d+),(\d+),(\d+),(\d+),(\d+)', response)
        if match:
            stat = int(match.group(3))
            states = {0: "connected", 1: "held", 2: "dialing",
                     3: "ringing", 4: "incoming", 5: "waiting"}
            return states.get(stat, "unknown")

        return "idle"  # No CLCC response means no active call

    def hangup(self):
        """End the call"""
        self._send_at("AT+CHUP")
```

---

## Module 2: Audio Router (audio_router.py)

### Critical: Anti-Aliasing Filter

When downsampling from 48kHz (USB sound card) to 24kHz (Whisper input), you MUST apply a low-pass filter first. Without this, high frequencies alias into audible garbage and speech recognition fails completely.

**WRONG (causes garbage transcriptions):**
```python
# Simple decimation - DO NOT USE
audio_24k = audio_48k[0::2]
```

**CORRECT:**
```python
import numpy as np

def resample_48k_to_24k(audio_48k: np.ndarray) -> np.ndarray:
    """Downsample with anti-aliasing filter"""
    audio_f = audio_48k.astype(np.float32)

    # Low-pass FIR filter to remove frequencies above 12kHz
    # (Nyquist frequency for 24kHz output)
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    filtered = np.convolve(audio_f, kernel, mode='same')

    # Then decimate
    decimated = filtered[0::2]

    return np.clip(decimated, -32768, 32767).astype(np.int16)

def resample_24k_to_48k(audio_24k: np.ndarray) -> np.ndarray:
    """Upsample with linear interpolation"""
    audio_f = audio_24k.astype(np.float32)
    resampled = np.zeros(len(audio_24k) * 2, dtype=np.float32)
    resampled[0::2] = audio_f
    resampled[1::2] = (audio_f + np.roll(audio_f, -1)) / 2
    resampled[-1] = audio_f[-1]
    return resampled.astype(np.int16)
```

### Audio Router Implementation

```python
import sounddevice as sd
import numpy as np
import queue
import threading

class AudioRouter:
    def __init__(self, device_name: str = "USB Advanced Audio Device"):
        self.device_name = device_name
        self.input_queue = queue.Queue()
        self.sample_rate = 48000  # USB device rate
        self.target_rate = 24000  # Whisper expects 16kHz or 24kHz
        self._running = False

    def find_device(self) -> int:
        """Find USB audio device index"""
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if self.device_name.lower() in d['name'].lower():
                return i
        raise ValueError(f"Device not found: {self.device_name}")

    def start(self):
        """Start audio capture"""
        device_idx = self.find_device()

        self.input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            device=device_idx,
            blocksize=1024,
            callback=self._audio_callback
        )

        self.output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            device=device_idx,
            blocksize=1024
        )

        self.input_stream.start()
        self.output_stream.start()
        self._running = True

    def _audio_callback(self, indata, frames, time, status):
        """Called when audio is captured"""
        audio = indata.flatten()
        audio_24k = self._resample_down(audio)
        self.input_queue.put(audio_24k.tobytes())

    def _resample_down(self, audio: np.ndarray) -> np.ndarray:
        """48kHz -> 24kHz with anti-aliasing"""
        audio_f = audio.astype(np.float32)
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
        filtered = np.convolve(audio_f, kernel, mode='same')
        decimated = filtered[0::2]
        return np.clip(decimated, -32768, 32767).astype(np.int16)

    def _resample_up(self, audio: np.ndarray) -> np.ndarray:
        """24kHz -> 48kHz with interpolation"""
        audio_f = audio.astype(np.float32)
        resampled = np.zeros(len(audio) * 2, dtype=np.float32)
        resampled[0::2] = audio_f
        resampled[1::2] = (audio_f + np.roll(audio_f, -1)) / 2
        resampled[-1] = audio_f[-1]
        return resampled.astype(np.int16)

    def read_audio(self) -> bytes:
        """Get audio chunk from queue"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def play_audio(self, audio_bytes: bytes):
        """Play audio to the call"""
        audio_24k = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_48k = self._resample_up(audio_24k)
        self.output_stream.write(audio_48k.reshape(-1, 1))

    def stop(self):
        """Stop audio streams"""
        self._running = False
        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()
        if hasattr(self, 'output_stream'):
            self.output_stream.stop()
            self.output_stream.close()
```

---

## Module 3: Speech-to-Text (stt.py)

### Option A: faster-whisper (Recommended)

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel
import numpy as np

class SpeechToText:
    def __init__(self, model_size: str = "base.en"):
        # Models: tiny.en, base.en, small.en, medium.en, large-v3
        # base.en is good balance of speed/accuracy
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.sample_rate = 16000  # Whisper expects 16kHz

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text"""
        # Convert bytes to float32 array normalized to [-1, 1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0

        # Transcribe
        segments, info = self.model.transcribe(
            audio_float,
            beam_size=5,
            language="en",
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Combine all segments
        text = " ".join(segment.text for segment in segments)
        return text.strip()
```

### Voice Activity Detection (VAD)

Use Silero VAD to detect when someone is speaking before sending to Whisper:

```bash
pip install silero-vad
```

```python
import torch

class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.5):
        self.model, utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', force_reload=False
        )
        self.get_speech_timestamps = utils[0]
        self.threshold = threshold
        self.sample_rate = 16000

    def is_speech(self, audio_bytes: bytes) -> bool:
        """Check if audio contains speech"""
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = torch.from_numpy(audio.astype(np.float32) / 32768.0)

        confidence = self.model(audio_float, self.sample_rate).item()
        return confidence > self.threshold
```

---

## Module 4: LLM Response Generation (llm.py)

### Option A: Claude Haiku (Cheap, High Quality)

```bash
pip install anthropic
```

```python
import anthropic

class LLMEngine:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.system_prompt = ""

    def set_objective(self, objective: str, context: dict):
        """Set the call objective and context"""
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        self.system_prompt = f"""You are an AI assistant making a phone call.

YOUR OBJECTIVE:
{objective}

CONTEXT:
{context_str}

RULES:
1. Be polite, professional, and natural-sounding
2. Keep responses brief (1-2 sentences) - this is a phone call
3. Listen and respond appropriately
4. If the objective is complete, say goodbye naturally
5. Don't use bullet points or formatting - just speak naturally

Respond with ONLY what you would say out loud. No narration or stage directions."""

        self.conversation_history = []

    def generate_response(self, user_text: str) -> str:
        """Generate AI response to user's speech"""
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest, fastest
            max_tokens=150,  # Keep responses short
            system=self.system_prompt,
            messages=self.conversation_history
        )

        assistant_text = response.content[0].text

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })

        return assistant_text
```

### Option B: Ollama (Free, Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:3b  # Small, fast
# or
ollama pull llama3.1:8b  # Better quality
```

```python
import requests

class OllamaLLM:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.conversation_history = []
        self.system_prompt = ""

    def set_objective(self, objective: str, context: dict):
        """Set the call objective"""
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        self.system_prompt = f"""You are an AI on a phone call.

OBJECTIVE: {objective}

CONTEXT:
{context_str}

Keep responses brief (1-2 sentences). Speak naturally."""
        self.conversation_history = []

    def generate_response(self, user_text: str) -> str:
        """Generate response using local Ollama"""
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history
                ],
                "stream": False,
                "options": {"num_predict": 100}
            }
        )

        assistant_text = response.json()["message"]["content"]

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })

        return assistant_text
```

---

## Module 5: Text-to-Speech (tts.py)

### Option A: Piper TTS (Recommended - High Quality, Fast)

```bash
pip install piper-tts
```

Download a voice model from https://github.com/rhasspy/piper/blob/master/VOICES.md

```python
from piper import PiperVoice
import wave
import io
import numpy as np

class TextToSpeech:
    def __init__(self, model_path: str = "en_US-lessac-medium.onnx"):
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = 22050  # Piper outputs 22050Hz

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (24kHz int16)"""
        # Generate audio
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            self.voice.synthesize(text, wav)

        # Read the audio
        audio_buffer.seek(44)  # Skip WAV header
        audio = np.frombuffer(audio_buffer.read(), dtype=np.int16)

        # Resample 22050 -> 24000
        audio_24k = self._resample(audio, self.sample_rate, 24000)

        return audio_24k.tobytes()

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio to target rate"""
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float32)).astype(np.int16)
```

### Option B: macOS say command (Simple, Built-in)

```python
import subprocess
import tempfile
import wave
import numpy as np

class MacTTS:
    def __init__(self, voice: str = "Samantha"):
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        """Use macOS say command to generate audio"""
        with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
            temp_path = f.name

        # Generate audio file
        subprocess.run([
            'say', '-v', self.voice, '-o', temp_path, text
        ], check=True)

        # Convert AIFF to WAV and read
        wav_path = temp_path.replace('.aiff', '.wav')
        subprocess.run([
            'ffmpeg', '-i', temp_path, '-ar', '24000', '-ac', '1',
            '-f', 's16le', wav_path, '-y'
        ], check=True, capture_output=True)

        with open(wav_path, 'rb') as f:
            audio_bytes = f.read()

        # Cleanup
        import os
        os.unlink(temp_path)
        os.unlink(wav_path)

        return audio_bytes
```

---

## Module 6: Main Agent (agent.py)

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class CallRequest:
    phone: str
    objective: str
    context: dict

@dataclass
class CallResult:
    success: bool
    transcript: list
    duration: float

class PhoneAgent:
    def __init__(self):
        self.modem = SIM7600Modem()
        self.audio = AudioRouter()
        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LLMEngine(api_key="your-api-key")  # or OllamaLLM()
        self.tts = TextToSpeech()

    async def make_call(self, request: CallRequest) -> CallResult:
        """Make a phone call and handle the conversation"""
        transcript = []
        start_time = time.time()

        # Setup
        if not self.modem.connect():
            raise Exception("Failed to connect to modem")

        self.llm.set_objective(request.objective, request.context)
        self.audio.start()

        # Dial
        if not self.modem.dial(request.phone):
            raise Exception("Failed to dial")

        # Wait for answer
        while True:
            state = self.modem.get_call_state()
            if state == "connected":
                break
            elif state == "idle":
                raise Exception("Call failed")
            await asyncio.sleep(0.5)

        print("Call connected!")

        # Main conversation loop
        audio_buffer = b""
        silence_start = None
        SILENCE_THRESHOLD = 0.8  # seconds of silence = end of utterance

        try:
            while True:
                # Check if call still active
                if self.modem.get_call_state() == "idle":
                    break

                # Read audio
                chunk = self.audio.read_audio()
                if chunk:
                    audio_buffer += chunk

                    # Check for speech
                    if self.vad.is_speech(chunk):
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_THRESHOLD:
                            # End of utterance - process it
                            if len(audio_buffer) > 16000:  # At least 0.5s of audio
                                # Transcribe
                                user_text = self.stt.transcribe(audio_buffer)

                                if user_text.strip():
                                    print(f"User: {user_text}")
                                    transcript.append({"role": "user", "content": user_text})

                                    # Generate response
                                    response_text = self.llm.generate_response(user_text)
                                    print(f"Agent: {response_text}")
                                    transcript.append({"role": "assistant", "content": response_text})

                                    # Speak response
                                    audio_response = self.tts.synthesize(response_text)
                                    self.audio.play_audio(audio_response)

                                    # Check for goodbye
                                    if any(word in response_text.lower() for word in ["goodbye", "bye", "take care"]):
                                        await asyncio.sleep(2)  # Let TTS finish
                                        break

                            audio_buffer = b""
                            silence_start = None

                await asyncio.sleep(0.01)

        finally:
            self.modem.hangup()
            self.audio.stop()

        return CallResult(
            success=True,
            transcript=transcript,
            duration=time.time() - start_time
        )


# Usage
async def main():
    agent = PhoneAgent()

    result = await agent.make_call(CallRequest(
        phone="5551234567",
        objective="You are calling from City Library to remind them their book is ready for pickup.",
        context={"library": "City Library", "book": "The Great Gatsby"}
    ))

    print(f"Call duration: {result.duration:.1f}s")
    for msg in result.transcript:
        print(f"{msg['role']}: {msg['content']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration (config.py)

```python
import os

# Modem settings
MODEM_VENDOR_ID = 0x1e0e
MODEM_PRODUCT_ID = 0x9001
MODEM_INTERFACE = 2

# Audio settings
AUDIO_DEVICE_NAME = "USB Advanced Audio Device"
DEVICE_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 24000

# LLM settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = "claude-3-haiku-20240307"
# Or for Ollama:
# OLLAMA_MODEL = "llama3.2:3b"

# STT settings
WHISPER_MODEL = "base.en"

# TTS settings
PIPER_MODEL_PATH = "en_US-lessac-medium.onnx"

# VAD settings
VAD_THRESHOLD = 0.5
SILENCE_DURATION = 0.8  # seconds

# Call settings
MAX_CALL_DURATION = 300  # 5 minutes
CALLS_DIR = "./calls"
```

---

## Dependencies (requirements.txt)

```
# USB communication
pyusb>=1.2.1

# Audio
sounddevice>=0.4.6
numpy>=1.24.0

# Speech-to-text
faster-whisper>=0.10.0
torch>=2.0.0  # For Silero VAD

# LLM (choose one)
anthropic>=0.18.0  # For Claude
# requests>=2.31.0  # For Ollama (usually pre-installed)

# Text-to-speech
piper-tts>=1.2.0

# Utilities
python-dotenv>=1.0.0
```

---

## Installation Steps

```bash
# 1. Create project
mkdir ai-phone-agent && cd ai-phone-agent

# 2. Create virtual environment (USE HOMEBREW PYTHON on Mac M-series!)
/opt/homebrew/bin/python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Piper voice model
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en_US-lessac-medium.tar.gz
tar -xzf voice-en_US-lessac-medium.tar.gz

# 5. Set environment variables
export ANTHROPIC_API_KEY="your-key-here"

# 6. Connect hardware
# - Plug in SIM7600G-H HAT via USB
# - Connect USB sound card
# - Wire audio cables as described above

# 7. Run
python agent.py
```

---

## Troubleshooting

### "No module named usb"
Use Homebrew Python on Mac M-series:
```bash
/opt/homebrew/bin/python3 -m pip install pyusb
```

### Modem not found
1. Check USB connection: `lsusb` or `system_profiler SPUSBDataType`
2. Look for SimTech vendor ID 0x1e0e

### No audio / garbled audio
1. Verify USB sound card is connected
2. Check cable routing (speaker→mic, headphone→mic)
3. Ensure AT+CSDVC=1 is sent before call
4. Verify anti-aliasing filter is applied during resampling

### Poor transcription quality
1. Check audio levels aren't clipping (use AT+CLVL=2)
2. Ensure anti-aliasing filter is applied when downsampling
3. Try a larger Whisper model (small.en instead of base.en)

### Call connects immediately (false detection)
CLCC parsing bug - ensure you're reading the stat field (position 2), not dir field (position 1)

---

## Cost Comparison

| Component | OpenAI Realtime API | New Architecture |
|-----------|--------------------|--------------------|
| STT | Included (~$0.06/min) | Local Whisper (FREE) |
| LLM | GPT-4 (~$0.30/min) | Haiku (~$0.001/call) or Ollama (FREE) |
| TTS | Included (~$0.06/min) | Local Piper (FREE) |
| **Total per 2-min call** | **~$0.80-1.00** | **~$0.002 or FREE** |

---

## Summary

This spec covers everything learned from building and debugging the phone agent:

1. **Hardware**: SIM7600G-H HAT + USB sound card + TRRS splitter
2. **Critical AT commands**: AT+CSDVC=1 (audio routing), AT+CLVL=2 (volume)
3. **CLCC parsing bug**: Parse stat field correctly (position 2)
4. **Anti-aliasing**: MUST filter before downsampling or transcription fails
5. **Cost optimization**: Local STT/TTS + cheap LLM = 99% cost reduction

The architecture is modular - each component can be swapped independently.
