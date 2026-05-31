"""
S.Y.N. — Listen Loop
======================
The main orchestration loop that ties everything together:
    Clap → Wake → Record → Transcribe → (eventually: Intent → Action)

ARCHITECTURE LESSON: THE EVENT LOOP PATTERN
───────────────────────────────────────────
This is the heart of SYN. It follows the "event loop" pattern:
    1. Wait for an event (clap detection)
    2. Handle the event (record + transcribe)
    3. Process the result (print for now, intent routing later)
    4. Go back to step 1

THREAD SAFETY:
    The clap detector runs its own loop. When it detects a clap,
    it calls on_wake() in a NEW thread. This means:
    - The clap detector keeps running (doesn't block)
    - But we need to be careful with shared state
    - Solution: we use a lock to prevent overlapping wake events
"""

import threading
from backend.voice.tts import speak, stop_speech, enqueue_speech, wait_speech_complete
from backend.voice.clap_detector import ClapDetector
from backend.voice.audio_recorder import AudioRecorder
from backend.voice.stt import SpeechToText
from backend.utils.logger import get_logger
from backend.utils.status import show_state, show_transcription
from backend.brain.intent_router import IntentRouter
from backend.brain.dispatcher import Dispatcher
import config

logger = get_logger("LOOP")


class ListenLoop:
    """
    Main SYN orchestrator — ties wake detection, recording, and STT together.
    """

    def __init__(self):
        self._detector = ClapDetector(on_wake=self._on_wake)
        self._recorder = AudioRecorder()
        self._stt = SpeechToText()
        self._intent_router = IntentRouter()
        self._dispatcher = Dispatcher()
        self._is_processing = False  # prevents overlapping wake events
        self._interrupted = False
        self._current_state = "READY"
        self._lock = threading.Lock()
        
    def _set_state(self, state: str):
        self._current_state = state
        show_state(state)
        # Notify the clap detector so it can adjust its noise threshold instantly
        if state == "RESPONDING":
            self._detector.set_assistant_speaking(True)
        else:
            self._detector.set_assistant_speaking(False)

    def start(self):
        """Start the main listen loop (blocking)."""
        logger.info("Listen loop starting...")
        self._set_state("READY")
        self._detector.start(blocking=True)

    def stop(self):
        """Stop the listen loop."""
        self._detector.stop()
        logger.info("Listen loop stopped.")

    def _on_wake(self):
        """
        Called when a clap is detected. This runs in a separate thread.

        FLOW:
            1. Speak greeting ("I'm listening")
            2. Record audio until silence
            3. Transcribe audio with Whisper
            4. Process the transcription
            5. Go back to listening for claps
        """
        with self._lock:
            if self._is_processing:
                if self._current_state == "RESPONDING":
                    logger.info("Interruption detected! Halting response.")
                    self._interrupted = True
                    stop_speech()
                return
            self._is_processing = True
            self._interrupted = False

        try:
            # Step 1: Greet
            stop_speech()
            self._set_state("LISTENING")
            speak(config.SYN_WAKE_GREETING)

            # Step 2: Record (blocks until user stops talking)
            wav_path = self._recorder.record()

            if wav_path is None:
                speak("I didn't hear anything. Try again.")
                self._set_state("READY")
                return

            # Step 3: Transcribe
            self._set_state("PROCESSING")
            result = self._stt.transcribe(wav_path)
            text = result.get("text", "").strip()

            # Step 4: Clean up the audio file
            AudioRecorder.cleanup(wav_path)

            # Step 5: Process the transcription
            if text:
                self._handle_command(text, result)
            else:
                speak("Sorry, I couldn't understand that.")

        except Exception as e:
            logger.error(f"Error during wake processing: {e}", exc_info=True)
            speak("Something went wrong. Try again.")

        finally:
            with self._lock:
                self._is_processing = False
            self._set_state("READY")

    def _handle_command(self, text: str, stt_result: dict):
        """
        Process a transcribed command through intent routing, streaming dispatching,
        and reading back responses sentence-by-sentence.
        """
        show_transcription(
            text,
            language=stt_result.get("language", ""),
            confidence=stt_result.get("confidence", 0),
        )

        # Detect user's language and tell TTS to lock voice for this response
        user_lang = stt_result.get("language", "en")
        from backend.voice import tts as tts_module
        tts_module.set_language_hint(user_lang)

        # 1. Route the query to identify intent
        route_result = self._intent_router.route(text)
        logger.info(f"Routed: \"{text}\" -> Intent: '{route_result.intent}' (Conf: {route_result.confidence})")

        # 2. Dispatch query and stream speech sentence-by-sentence
        stop_speech()
        self._set_state("RESPONDING")
        print()
        print("  SYN >> ", end="", flush=True)
        
        chunk_generator = self._dispatcher.dispatch_stream(route_result)
        
        sentence_buffer = ""
        # Hindi uses '।' (Devanagari danda) as a full stop
        sentence_endings = {'.', '?', '!', '।'}
        full_response = []
        
        for chunk in chunk_generator:
            if self._interrupted:
                break
                
            # Print chunk live to the console
            print(chunk, end="", flush=True)
            full_response.append(chunk)
            sentence_buffer += chunk
            
            # Check for completed sentences to speak immediately
            while True:
                end_idx = -1
                ending_len = 0
                for char in sentence_endings:
                    # Look for punctuation followed by space
                    idx = sentence_buffer.find(char + " ")
                    if idx != -1:
                        if end_idx == -1 or idx < end_idx:
                            end_idx = idx
                            ending_len = 2 # length of char + space
                    # Also look for punctuation followed by newline
                    idx = sentence_buffer.find(char + "\n")
                    if idx != -1:
                        if end_idx == -1 or idx < end_idx:
                            end_idx = idx
                            ending_len = 2
                            
                # Check for raw newline boundaries
                if end_idx == -1:
                    idx = sentence_buffer.find("\n")
                    if idx != -1:
                        end_idx = idx
                        ending_len = 1
                        
                if end_idx == -1:
                    break
                    
                # Extract the complete sentence
                sentence = sentence_buffer[:end_idx + 1].strip()
                sentence_buffer = sentence_buffer[end_idx + ending_len:]
                
                # Clean up punctuation that breaks TTS (like unmatched quotes, markdown)
                clean_sentence = sentence.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
                if clean_sentence:
                    enqueue_speech(clean_sentence)
                    
        if not self._interrupted:
            # Speak any remaining words in buffer
            leftover = sentence_buffer.strip()
            clean_leftover = leftover.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
            if clean_leftover:
                enqueue_speech(clean_leftover)
                
            print() # Newline at the end
            print()
            
            wait_speech_complete()
        else:
            print("\n  [Response Interrupted]")
            logger.info("Pipeline was interrupted by user.")
        
        # Reset language hint after response completes
        tts_module.set_language_hint(None)
        
        final_text = "".join(full_response).strip()
        logger.info(f"Pipeline executed successfully. Spoke: \"{final_text}\"")
