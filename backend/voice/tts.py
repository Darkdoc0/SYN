import os
import tempfile
import threading
import ctypes
import queue
import re
import asyncio
import pyttsx3
import config
from backend.utils.logger import get_logger

logger = get_logger("TTS")

_thread_local = threading.local()

def _get_engine():
    """Lazily initializes the pyttsx3 engine for the current active thread."""
    if not hasattr(_thread_local, "engine"):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
            
        engine = pyttsx3.init()
        engine.setProperty("rate", config.TTS_RATE)
        engine.setProperty("volume", config.TTS_VOLUME)
        
        voices = engine.getProperty("voices")
        if config.TTS_VOICE_INDEX < len(voices):
            engine.setProperty("voice", voices[config.TTS_VOICE_INDEX].id)
            
        _thread_local.engine = engine
        
    return _thread_local.engine

def play_mp3_native(file_path: str):
    """Plays an MP3 file using Windows Multi-Media Interface (MCI) natively."""
    try:
        short_path_buffer = ctypes.create_unicode_buffer(260)
        ctypes.windll.kernel32.GetShortPathNameW(file_path, short_path_buffer, 260)
        short_path = short_path_buffer.value
        
        alias = f"syn_tts_{id(file_path)}_{int(ctypes.windll.kernel32.GetTickCount())}"
        ctypes.windll.winmm.mciSendStringW(f"open {short_path} type mpegvideo alias {alias}", None, 0, 0)
        ctypes.windll.winmm.mciSendStringW(f"play {alias} wait", None, 0, 0)
        ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, 0)
    except Exception as e:
        logger.error(f"Native Windows MCI MP3 playback crashed: {e}")

def is_hindi(text: str) -> bool:
    """Detects if text contains Devanagari characters."""
    return bool(re.search(r'[\u0900-\u097f]', text))

class TTSQueueManager:
    """Manages background queues for synthesizing and playing speech seamlessly."""
    
    def __init__(self):
        self.synthesis_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        self.current_session_id = 0
        self.lock = threading.Lock()
        self._language_hint = None  # "hi" or "en" or None (auto-detect)
        
        self.synth_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        self.play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.synth_thread.start()
        self.play_thread.start()

    def set_language_hint(self, lang: str | None):
        """Lock the voice to a specific language for the duration of a response.
        Set to None to go back to auto-detection."""
        self._language_hint = lang
        if lang:
            logger.info(f"Voice locked to: English (SoniaNeural) - Hindi voice disabled by user")

    def _pick_voice(self, text: str) -> str:
        """Always return the default English voice (Sonia) as requested by user."""
        return getattr(config, "TTS_EDGE_VOICE_EN", "en-GB-SoniaNeural")

    def _synthesis_worker(self):
        while True:
            session_id, text = self.synthesis_queue.get()
            if session_id != self.current_session_id:
                self.synthesis_queue.task_done()
                continue
                
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"syn_stream_{session_id}_{id(text)}.mp3")
            
            success = False
            try:
                engine_choice = getattr(config, "TTS_ENGINE", "edge-tts").lower()
                if engine_choice == "edge-tts":
                    import edge_tts
                    voice = self._pick_voice(text)
                    logger.debug(f"Synthesizing with voice '{voice}': \"{text[:50]}...\"")
                    async def _synth():
                        communicate = edge_tts.Communicate(text, voice)
                        await communicate.save(temp_file)
                    asyncio.run(_synth())
                    success = True
                elif engine_choice == "gtts":
                    from gtts import gTTS
                    use_hindi = self._language_hint == "hi" or (self._language_hint is None and is_hindi(text))
                    tts = gTTS(text=text, lang='hi' if use_hindi else 'en', tld='co.in' if use_hindi else 'co.uk')
                    tts.save(temp_file)
                    success = True
            except Exception as e:
                logger.error(f"Online synthesis failed: {e}. Falling back to pyttsx3.")
                
            if success:
                self.playback_queue.put((session_id, text, temp_file, False))
            else:
                self.playback_queue.put((session_id, text, None, True))
                
            self.synthesis_queue.task_done()

    def _playback_worker(self):
        while True:
            session_id, text, temp_file, use_offline = self.playback_queue.get()
            if session_id != self.current_session_id:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                self.playback_queue.task_done()
                continue
                
            if use_offline or temp_file is None:
                try:
                    engine = _get_engine()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    logger.error(f"Local pyttsx3 failed: {e}")
            else:
                play_mp3_native(temp_file)
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
            self.playback_queue.task_done()

    def stop_speech(self):
        """Stops active playback and clears queues."""
        with self.lock:
            self.current_session_id += 1
            
        try:
            ctypes.windll.winmm.mciSendStringW("stop all", None, 0, 0)
            ctypes.windll.winmm.mciSendStringW("close all", None, 0, 0)
        except Exception as e:
            logger.error(f"Error stopping MCI playback: {e}")
            
        # Drain queues
        while not self.synthesis_queue.empty():
            try:
                self.synthesis_queue.get_nowait()
                self.synthesis_queue.task_done()
            except queue.Empty:
                break
                
        while not self.playback_queue.empty():
            try:
                session_id, text, temp_file, use_offline = self.playback_queue.get_nowait()
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                self.playback_queue.task_done()
            except queue.Empty:
                break

    def enqueue_speech(self, text: str):
        # Phonetic adjustments for voice engine
        phonetic_text = text.replace("S.Y.N.", "Sin").replace("S.Y.N", "Sin")
        phonetic_text = re.sub(r'\bSYN\b', 'Sin', phonetic_text)
        
        with self.lock:
            self.synthesis_queue.put((self.current_session_id, phonetic_text))

    def wait_speech_complete(self):
        self.synthesis_queue.join()
        self.playback_queue.join()

_manager = TTSQueueManager()

def speak(text: str):
    """Synchronous speech. Used for short system messages."""
    print(f"  {config.SYN_NAME}: {text}")
    _manager.stop_speech()
    _manager.enqueue_speech(text)
    _manager.wait_speech_complete()

def enqueue_speech(text: str):
    """Asynchronous/Queue-based speech. Used for streaming LLM responses."""
    _manager.enqueue_speech(text)

def stop_speech():
    """Immediately halts any speech playback and clears the queue."""
    _manager.stop_speech()

def wait_speech_complete():
    """Blocks until all queued speech has finished playing."""
    _manager.wait_speech_complete()

def set_language_hint(lang: str | None):
    """Set the language hint for the TTS voice selection."""
    _manager.set_language_hint(lang)

def list_voices():
    """Print all available offline system voices."""
    try:
        engine = _get_engine()
        voices = engine.getProperty("voices")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name} — {voice.id}")
    except Exception:
        pass