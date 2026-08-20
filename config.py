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
MIC_SAMPLE_RATE = 16000          # Hz — standard 16kHz for Whisper & openWakeWord
MIC_CHUNK_SIZE = 1280            # 80ms chunk size for openWakeWord
MIC_FORMAT_WIDTH = 2             # Bytes per sample (2 = 16-bit audio)
MIC_CHANNELS = 1                 # 1 = mono
SPEECH_ENERGY_THRESHOLD = 400    # Audio energy required to trigger recording

# ──────────────────────────────────────────────
#  WAKE WORD ENGINE (openWakeWord)
# ──────────────────────────────────────────────
WAKE_WORD_MODEL = "hey_syn"      # "hey_syn" (pronounced 'sin'), "hey_jarvis", "alexa"
WAKE_WORD_THRESHOLD = 0.5        # Detection confidence (0.0 to 1.0)
CONVERSATION_WINDOW = 15.0       # Seconds — S.Y.N. stays 'awake' for follow-up questions

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
STT_ENGINE = "whisper"           # "whisper" (offline) or "google" (cloud fallback)
STT_WHISPER_MODEL = "small"      # tiny, base, small, medium, large
STT_WHISPER_DEVICE = "cpu"       # "cuda" for GPU, "cpu" otherwise (fallback to CPU to avoid CUDA hangs)
STT_WHISPER_COMPUTE = "int8"     # "float16" for GPU, "int8" for CPU
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
