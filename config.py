"""
S.Y.N. Central Configuration
=============================
All tunable settings in one place. Adjust these to match your
hardware (mic sensitivity, clap thresholds, etc.)
"""

# ──────────────────────────────────────────────
#  MICROPHONE
# ──────────────────────────────────────────────
MIC_DEVICE_INDEX = None          # None = system default mic. Set to int for specific device.
MIC_SAMPLE_RATE = 44100          # Hz — standard audio sample rate
MIC_CHUNK_SIZE = 1024            # Frames per buffer read (lower = more responsive, higher CPU)
MIC_FORMAT_WIDTH = 2             # Bytes per sample (2 = 16-bit audio)
MIC_CHANNELS = 1                 # 1 = mono (clap detection only needs mono)

# ──────────────────────────────────────────────
#  CLAP DETECTION
# ──────────────────────────────────────────────
CLAP_ENERGY_THRESHOLD = 1000     # Calibrated for your room (peak noise ~201)
CLAP_MIN_FREQUENCY = 1500        # Hz — claps are typically broadband, above 1.5 kHz
CLAP_MAX_FREQUENCY = 8000        # Hz — upper bound for clap frequency detection
CLAP_MIN_DURATION_MS = 5         # Min spike duration in ms (claps are very short)
CLAP_MAX_DURATION_MS = 120       # Max spike duration in ms
CLAP_DOUBLE_TAP_WINDOW = 1.2    # Seconds — max time between two claps for a "double clap"
CLAP_COOLDOWN = 2.0              # Seconds — ignore claps after a successful double-clap
CLAP_PATTERN = "double"          # "single" or "double" — trigger mode

# ──────────────────────────────────────────────
#  TTS (Text-to-Speech)
# ──────────────────────────────────────────────
TTS_ENGINE = "edge-tts"          # "edge-tts" (online neural), "pyttsx3" (offline), or "gtts"
TTS_EDGE_VOICE_EN = "en-GB-SoniaNeural"  # Default English Voice
TTS_EDGE_VOICE_HI = "hi-IN-SwaraNeural"  # Default Hindi Voice
TTS_RATE = 175                   # Words per minute (pyttsx3)
TTS_VOLUME = 1.0                 # 0.0 to 1.0
TTS_VOICE_INDEX = 0              # 0 = default voice, 1 = alt voice (varies by OS)

# ──────────────────────────────────────────────
#  STT (Speech-to-Text) — Day 3
# ──────────────────────────────────────────────
STT_ENGINE = "google"            # "whisper" (offline) or "google" (cloud fallback)
STT_WHISPER_MODEL = "base"       # tiny, base, small, medium, large
STT_SILENCE_TIMEOUT = 2.0        # Seconds of silence before stopping recording
STT_MAX_RECORD_TIME = 20.0       # Max recording time in seconds

# ──────────────────────────────────────────────
#  LLM / BRAIN — Day 6
# ──────────────────────────────────────────────
LLM_PROVIDER = "ollama"          # "ollama" (offline) or "openai" or "anthropic"
LLM_MODEL = "llama3"             # Model name
LLM_TEMPERATURE = 0.7            # Creativity (0.0 = factual, 1.0 = creative)
LLM_MAX_TOKENS = 500             # Max response length

# ──────────────────────────────────────────────
#  SYSTEM
# ──────────────────────────────────────────────
SYN_NAME = "SYN"
SYN_WAKE_GREETING = "I'm listening."
SYN_BOOT_MESSAGE = "Initializing S.Y.N. Synthetic Yielding Nexus is online."
LOG_FILE = "logs/syn.log"
DEBUG_MODE = True                 # Extra console output for development
