"""
S.Y.N. — Speech-to-Text Engine
================================
Converts audio (WAV files) into text using Whisper (offline) with
Google Speech Recognition as a cloud fallback.

HOW WHISPER WORKS (learn this!):
────────────────────────────────
Whisper is a neural network trained on 680,000 hours of speech.

    Audio → Mel Spectrogram → Transformer → Text

Step 1: MEL SPECTROGRAM
    Raw audio is a 1D wave (amplitude over time).
    A mel spectrogram converts it to 2D: time × frequency.
    "Mel" = a frequency scale that matches how humans hear.
    Low frequencies get more detail (we're more sensitive there).
    
    Think of it as taking the audio and turning it into an image
    where each pixel's brightness = how loud that frequency is at that moment.

Step 2: TRANSFORMER
    Same architecture as ChatGPT, but for audio → text.
    It reads the spectrogram left to right and predicts words.
    It was trained on audio from YouTube, podcasts, audiobooks, etc.

Step 3: TEXT OUTPUT
    The model outputs tokens (word pieces) that form the transcription.
    It also detects the language automatically.

MODEL SIZES:
    tiny   = 39M params  → fast, less accurate     (~1 GB RAM)
    base   = 74M params  → good balance             (~1 GB RAM)  ← we use this
    small  = 244M params → better accuracy           (~2 GB RAM)
    medium = 769M params → very accurate             (~5 GB RAM)
    large  = 1.5B params → best accuracy             (~10 GB RAM)

WHY faster-whisper?
    Original Whisper (by OpenAI) is in PyTorch — slow on CPU.
    faster-whisper uses CTranslate2 — an optimized inference engine.
    Same model, same accuracy, but 4x faster and uses less memory.

FALLBACK: GOOGLE SPEECH RECOGNITION
    If Whisper isn't installed or fails, we fall back to Google's
    free speech-to-text API (requires internet). It's less private
    but works as a backup.
"""

import os
import config
from backend.utils.logger import get_logger

logger = get_logger("STT")


# ──────────────────────────────────────────────
#  Whisper Engine (Offline — Primary)
# ──────────────────────────────────────────────
class WhisperSTT:
    """
    Offline speech-to-text using faster-whisper.
    
    The model is downloaded once (~150 MB for 'base') and cached locally.
    After that, it works completely offline — no internet needed.
    """

    def __init__(self, model_size: str = None):
        """
        Args:
            model_size: Whisper model to use. Options:
                       "tiny", "base", "small", "medium", "large-v3"
                       Bigger = more accurate but slower.
        """
        self.model_size = model_size or config.STT_WHISPER_MODEL
        self._model = None  # lazy loaded (only loads when first used)

    def _load_model(self):
        """
        Load the Whisper model into memory.
        
        WHY LAZY LOADING?
            The model takes a few seconds to load and uses ~1 GB RAM.
            We don't want to pay this cost at boot if the user
            hasn't spoken yet. So we load it the first time
            transcribe() is called, then keep it in memory.
        """
        if self._model is not None:
            return  # already loaded

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model '{self.model_size}'...")
            logger.info("(First time will download ~150 MB. After that, it's cached.)")

            # compute_type explanation:
            #   "int8"  = 8-bit integers — fastest on CPU, slightly less accurate
            #   "float16" = half precision — needs GPU (CUDA)
            #   "float32" = full precision — slowest but most accurate on CPU
            # We use int8 for speed on CPU. If you have an NVIDIA GPU,
            # change device to "cuda" and compute_type to "float16".
            self._model = WhisperModel(
                self.model_size,
                device="cpu",           # "cpu" or "cuda" (GPU)
                compute_type="int8",    # quantization for CPU speed
            )

            logger.info("Whisper model loaded successfully!")

        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. Run:\n"
                "  pip install faster-whisper\n"
                "Or set STT_ENGINE='google' in config.py to use cloud STT."
            )

    def transcribe(self, wav_path: str) -> dict:
        """
        Transcribe a WAV audio file to text.

        Args:
            wav_path: Path to a .wav file

        Returns:
            dict with:
                - "text": The transcribed text (string)
                - "language": Detected language code (e.g., "en")
                - "confidence": Average confidence score (0.0 to 1.0)
                - "segments": List of timed segments (for subtitles, etc.)

        HOW THE OUTPUT WORKS:
            Whisper returns "segments" — chunks of text with timestamps.
            Example:
                [0.0s - 2.5s] "Hey SYN, what's the weather"
                [2.5s - 4.0s] "like today?"
            We join them into one string for the intent router.
        """
        self._load_model()

        if not os.path.exists(wav_path):
            return {"text": "", "language": "", "confidence": 0.0, "segments": []}

        logger.info(f"Transcribing: {wav_path}")

        # Transcribe!
        # beam_size: how many hypotheses to consider at each step.
        #   Higher = more accurate but slower. 5 is the sweet spot.
        # vad_filter: Voice Activity Detection — skips silent parts.
        #   Makes transcription faster by not processing silence.
        segments, info = self._model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True,       # skip silence for speed
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 0.5s silence = pause
            ),
        )

        # Collect all segments
        all_segments = []
        full_text_parts = []

        for segment in segments:
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts).strip()

        # Calculate average confidence (if available)
        # Whisper doesn't give per-word confidence in faster-whisper,
        # but we can use the language probability as a proxy.
        confidence = round(info.language_probability, 3)

        result = {
            "text": full_text,
            "language": info.language,
            "confidence": confidence,
            "segments": all_segments,
        }

        logger.info(f"Result: \"{full_text}\"")
        logger.info(f"Language: {info.language} (confidence: {confidence})")

        return result


# ──────────────────────────────────────────────
#  Google Speech Recognition (Online — Fallback)
# ──────────────────────────────────────────────
class GoogleSTT:
    """
    Cloud-based speech-to-text using Google's free API.
    
    WHY A FALLBACK?
        - Whisper needs ~1 GB RAM and a few seconds to load
        - Google STT is instant but needs internet
        - Google's free tier has rate limits (not for heavy use)
        - Less private (audio is sent to Google's servers)
    
    We use the `speech_recognition` library which wraps Google's API.
    It handles audio format conversion automatically.
    """

    def __init__(self):
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
        except ImportError:
            raise ImportError(
                "speech_recognition is not installed. Run:\n"
                "  pip install SpeechRecognition"
            )

    def transcribe(self, wav_path: str) -> dict:
        """
        Transcribe using Google's cloud STT with bilingual support.
        Uses a single API call — fast and reliable.
        """
        import speech_recognition as sr
        import re

        if not os.path.exists(wav_path):
            return {"text": "", "language": "en", "confidence": 0.0, "segments": []}

        logger.info(f"Transcribing via Google: {wav_path}")

        try:
            with sr.AudioFile(wav_path) as source:
                audio = self._recognizer.record(source)

            # Single call with Hindi — Google will still transcribe English spoken words
            # correctly when using hi-IN, but will also catch Hindi. This avoids
            # the slow double-call approach.
            try:
                text = self._recognizer.recognize_google(audio, language="hi-IN")
            except sr.UnknownValueError:
                # Hindi recognizer couldn't parse — try English
                try:
                    text = self._recognizer.recognize_google(audio, language="en-IN")
                except sr.UnknownValueError:
                    logger.warning("Google could not understand audio in any language.")
                    return {"text": "", "language": "en", "confidence": 0.0, "segments": []}
            
            # Detect language from the actual text content
            has_devanagari = bool(re.search(r'[\u0900-\u097f]', text))
            detected_lang = "hi" if has_devanagari else "en"

            logger.info(f"Google result: \"{text}\" [lang={detected_lang}]")

            return {
                "text": text,
                "language": detected_lang,
                "confidence": 0.9,
                "segments": [],
            }

        except sr.UnknownValueError:
            # Google couldn't understand the audio
            logger.warning("Google could not understand audio.")
            return {"text": "", "language": "en", "confidence": 0.0, "segments": []}

        except sr.RequestError as e:
            # Network error
            logger.error(f"Google API error: {e}")
            return {"text": "", "language": "en", "confidence": 0.0, "segments": []}


# ──────────────────────────────────────────────
#  Unified STT Interface
# ──────────────────────────────────────────────
class SpeechToText:
    """
    Unified speech-to-text interface.
    
    Automatically selects engine based on config:
        - "whisper" → WhisperSTT (offline, private, accurate)
        - "google"  → GoogleSTT (online, fast, simple)
    
    Falls back to Google if Whisper fails.

    DESIGN PATTERN: STRATEGY PATTERN
        This is a common pattern in software engineering.
        Instead of hardcoding one STT engine, we define an interface
        (transcribe) and swap implementations behind it.
        The rest of our code just calls stt.transcribe() and doesn't
        care whether it's Whisper or Google underneath.
    """

    def __init__(self):
        self.engine_name = config.STT_ENGINE

        if self.engine_name == "whisper":
            try:
                self._primary = WhisperSTT()
                self._fallback = None  # will create Google on demand
                logger.info("Engine: Whisper (offline)")
            except ImportError:
                logger.warning("faster-whisper not installed. Using Google fallback.")
                self._primary = GoogleSTT()
                self._fallback = None

        elif self.engine_name == "google":
            self._primary = GoogleSTT()
            self._fallback = None
            logger.info("Engine: Google (online)")

        else:
            raise ValueError(f"Unknown STT engine: {self.engine_name}")

    def transcribe(self, wav_path: str) -> dict:
        """
        Transcribe a WAV file to text.
        
        Tries primary engine first. If it fails, falls back to secondary.
        
        Returns:
            dict with "text", "language", "confidence", "segments"
        """
        try:
            result = self._primary.transcribe(wav_path)

            # If primary returned empty text, try fallback
            if not result["text"] and self._fallback is not None:
                logger.info("Primary returned empty. Trying fallback...")
                result = self._fallback.transcribe(wav_path)

            return result

        except Exception as e:
            logger.error(f"Primary engine error: {e}")

            # Try fallback
            if self._fallback is not None:
                try:
                    return self._fallback.transcribe(wav_path)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            return {"text": "", "language": "", "confidence": 0.0, "segments": []}


# ──────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from backend.voice.audio_recorder import AudioRecorder

    print("=" * 55)
    print("  S.Y.N. -- Speech-to-Text Test")
    print("  Speak after the prompt, then go silent.")
    print("  Your speech will be transcribed by Whisper.")
    print("=" * 55)
    print()

    # Step 1: Record
    recorder = AudioRecorder()
    print("Say something...")
    wav_file = recorder.record()

    if wav_file:
        # Step 2: Transcribe
        stt = SpeechToText()
        result = stt.transcribe(wav_file)

        print()
        print("-" * 40)
        print(f"  You said: \"{result['text']}\"")
        print(f"  Language: {result['language']}")
        print(f"  Confidence: {result['confidence']}")
        print("-" * 40)

        # Cleanup temp file
        AudioRecorder.cleanup(wav_file)
    else:
        print("No speech detected.")
