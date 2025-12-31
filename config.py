import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Anthropic (for Claude Haiku)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# =============================================================================
# SIM7600 HAT Audio Configuration
# =============================================================================
# The SIM7600G-H HAT (B) has a single TRRS 3.5mm combo jack.
# You need a TRRS splitter + USB sound card to route audio properly:
#   - HAT speaker output → USB sound card mic input (captures call audio)
#   - USB sound card headphone output → HAT mic input (sends AI voice)
#
# Set these to your USB sound card device names:
SIM7600_AUDIO_INPUT = os.getenv("SIM7600_AUDIO_INPUT", "USB Advanced Audio Device")  # Receives HAT speaker
SIM7600_AUDIO_OUTPUT = os.getenv("SIM7600_AUDIO_OUTPUT", "USB Advanced Audio Device")  # Sends to HAT mic

# =============================================================================
# FaceTime/Continuity Audio (Original - not recommended)
# =============================================================================
# Note: FaceTime Continuity bypasses Mac audio system, making this unreliable.
# Use SIM7600 HAT instead for reliable AI phone calls.
AUDIO_INPUT_DEVICE = "iPhone"           # Captures call audio from iPhone
AUDIO_OUTPUT_DEVICE = "MacBook Pro Speakers"  # AI voice plays on speakers → picked up by iPhone mic
SYSTEM_MIC = "MacBook Pro Microphone"  # Your actual mic (for testing)
SYSTEM_SPEAKER = "MacBook Pro Speakers"  # Your actual speaker (for testing)

# Audio settings
SAMPLE_RATE = 24000  # OpenAI Realtime API uses 24kHz
CHANNELS = 1
CHUNK_SIZE = 1024

# OpenAI Realtime API settings
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"
VOICE = "alloy"  # Options: alloy, echo, shimmer, ash, ballad, coral, sage, verse

# Call settings
MAX_CALL_DURATION = 600  # 10 minutes max
HOLD_DETECTION_THRESHOLD = 30  # seconds of music before considering "on hold"

# Paths
CALLS_DIR = os.path.join(os.path.dirname(__file__), "calls")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
