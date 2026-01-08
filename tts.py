"""
Text-to-Speech using Piper TTS

Converts text to natural-sounding speech using local neural TTS.
"""

import numpy as np
import logging
import wave
import io
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _digits_to_words(digits: str) -> str:
    """Convert a string of digits to spoken words (digit by digit)"""
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    return ' '.join(digit_words.get(d, d) for d in digits if d.isdigit())


def _phone_to_words(phone: str) -> str:
    """Convert phone number to naturally spoken words"""
    # Extract just digits
    digits = ''.join(c for c in phone if c.isdigit())

    if len(digits) == 10:
        # Format: XXX XXX XXXX with pauses
        return f"{_digits_to_words(digits[:3])}, {_digits_to_words(digits[3:6])}, {_digits_to_words(digits[6:])}"
    elif len(digits) == 11 and digits[0] == '1':
        # Format: 1 XXX XXX XXXX
        return f"one, {_digits_to_words(digits[1:4])}, {_digits_to_words(digits[4:7])}, {_digits_to_words(digits[7:])}"
    else:
        # Just read digits with pauses every 3-4 digits
        return _digits_to_words(digits)


def _card_to_words(card: str) -> str:
    """Convert credit card number to spoken words (groups of 4)"""
    digits = ''.join(c for c in card if c.isdigit())
    groups = [digits[i:i+4] for i in range(0, len(digits), 4)]
    return ', '.join(_digits_to_words(g) for g in groups)


def _number_to_words(n: int) -> str:
    """Convert integer to words"""
    if n == 0:
        return "zero"

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n < 0:
        return "negative " + _number_to_words(-n)
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else " " + ones[n % 10])
    if n < 1000:
        return ones[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _number_to_words(n % 100))
    if n < 1000000:
        return _number_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _number_to_words(n % 1000))
    if n < 1000000000:
        return _number_to_words(n // 1000000) + " million" + ("" if n % 1000000 == 0 else " " + _number_to_words(n % 1000000))
    return str(n)  # Fallback for very large numbers


def preprocess_text_for_speech(text: str) -> str:
    """
    Preprocess text to be more naturally spoken by TTS.

    Handles:
    - Phone numbers (read digit by digit)
    - Credit card numbers (read digit by digit in groups of 4)
    - Currency ($100.40 -> one hundred dollars and forty cents)
    - Ranges (10-15 -> ten to fifteen)
    - Percentages (50% -> fifty percent)
    - Common symbols
    - Abbreviations
    """
    if not text:
        return text

    result = text

    # Phone numbers: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX, +1XXXXXXXXXX
    result = re.sub(
        r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        lambda m: _phone_to_words(m.group(0)),
        result
    )

    # Credit card numbers: XXXX XXXX XXXX XXXX or XXXX-XXXX-XXXX-XXXX or 16 digits
    result = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        lambda m: _card_to_words(m.group(0)),
        result
    )

    # CVV/CVC: 3-4 digit codes (when preceded by CVV, CVC, security code, etc.)
    result = re.sub(
        r'\b(CVV|CVC|security code|code)[:\s]*(\d{3,4})\b',
        lambda m: m.group(1) + ' ' + _digits_to_words(m.group(2)),
        result,
        flags=re.IGNORECASE
    )

    # Currency: $X.XX or $X
    def replace_currency(match):
        amount = match.group(1).replace(",", "")
        if "." in amount:
            dollars, cents = amount.split(".")
            dollars = int(dollars) if dollars else 0
            cents = int(cents.ljust(2, '0')[:2]) if cents else 0

            if cents == 0:
                return _number_to_words(dollars) + " dollars"
            elif dollars == 0:
                return _number_to_words(cents) + " cents"
            else:
                return _number_to_words(dollars) + " dollars and " + _number_to_words(cents) + " cents"
        else:
            return _number_to_words(int(amount)) + " dollars"

    result = re.sub(r'\$([0-9,]+\.?[0-9]*)', replace_currency, result)

    # Ranges with hyphen: 10-15 -> ten to fifteen (but not negative numbers)
    def replace_range(match):
        # Check if it's likely a range (preceded by space or start, followed by space/word)
        num1, num2 = match.group(1), match.group(2)
        return _number_to_words(int(num1)) + " to " + _number_to_words(int(num2))

    result = re.sub(r'\b(\d+)-(\d+)\b', replace_range, result)

    # Percentages: 50% -> fifty percent
    def replace_percent(match):
        num = match.group(1).replace(",", "")
        if "." in num:
            return num + " percent"  # Keep decimal percentages as-is
        return _number_to_words(int(num)) + " percent"

    result = re.sub(r'(\d+(?:\.\d+)?)\s*%', replace_percent, result)

    # Times: 2:30 -> two thirty
    def replace_time(match):
        hour, minute = int(match.group(1)), match.group(2)
        suffix = match.group(3) if match.group(3) else ""

        if minute == "00":
            time_str = _number_to_words(hour) + " o'clock"
        else:
            time_str = _number_to_words(hour) + " " + _number_to_words(int(minute))

        if suffix:
            # "pm" -> "p.m."
            time_str += " " + suffix[0].lower() + "." + suffix[1].lower() + "."

        return time_str

    result = re.sub(r'\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM|a\.m\.|p\.m\.)?', replace_time, result)

    # Common symbols
    result = result.replace(" & ", " and ")
    result = result.replace("&", " and ")
    result = re.sub(r'\s*@\s*', " at ", result)
    result = result.replace(" + ", " plus ")
    result = result.replace(" = ", " equals ")
    result = result.replace(" / ", " or ")
    result = result.replace("24/7", "twenty-four seven")

    # Abbreviations
    result = re.sub(r'\bDr\.\s', "Doctor ", result)
    result = re.sub(r'\bMr\.\s', "Mister ", result)
    result = re.sub(r'\bMrs\.\s', "Missus ", result)
    result = re.sub(r'\bMs\.\s', "Miss ", result)
    result = re.sub(r'\bSt\.\s', "Street ", result)
    result = re.sub(r'\bAve\.\s', "Avenue ", result)
    result = re.sub(r'\bBlvd\.\s', "Boulevard ", result)
    result = re.sub(r'\betc\.', "et cetera", result)
    result = re.sub(r'\be\.g\.', "for example", result)
    result = re.sub(r'\bi\.e\.', "that is", result)

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result

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
            # Preprocess text for natural speech
            processed_text = preprocess_text_for_speech(text)
            if processed_text != text:
                logger.debug(f"TTS preprocessing: '{text[:50]}' -> '{processed_text[:50]}'")

            # Synthesize using generator
            audio_bytes_list = []
            for chunk in self.voice.synthesize(processed_text):
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
