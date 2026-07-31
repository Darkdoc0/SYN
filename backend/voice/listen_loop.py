"""
S.Y.N. — Listen Loop (JARVIS Mode)
==================================
The main orchestration loop for the Always-Listening Context Engine.
Continuously records, transcribes, and uses intelligent filtering
to decide whether to respond.
"""

import threading
import time
from backend.voice.tts import speak, stop_speech, enqueue_speech, wait_speech_complete
from backend.voice.audio_recorder import AudioRecorder
from backend.voice.stt import SpeechToText
from backend.voice.context_filter import ContextFilter
from backend.utils.logger import get_logger
from backend.utils.status import show_state, show_transcription
from backend.brain.intent_router import IntentRouter
from backend.brain.dispatcher import Dispatcher
import config

logger = get_logger("LOOP")


class ListenLoop:
    """
    Main SYN orchestrator — Always listening, transcribing, and routing.
    """

    def __init__(self):
        self._recorder = AudioRecorder()
        self._stt = SpeechToText()
        self._intent_router = IntentRouter()
        self._dispatcher = Dispatcher()
        self._context_filter = ContextFilter()
        
        self._is_processing = False
        self._interrupted_loop = False
        self._interrupted_response = False
        self._current_state = "READY"
        self._lock = threading.Lock()
        
    def _set_state(self, state: str):
        self._current_state = state
        show_state(state)

    def start(self):
        """Start the main always-listening loop (blocking)."""
        logger.info("Listen loop starting in JARVIS Mode...")
        self._set_state("READY")
        self._interrupted_loop = False
        
        # Greet on boot
        speak(config.SYN_BOOT_MESSAGE)

        while not self._interrupted_loop:
            try:
                self._set_state("LISTENING")
                
                # Blocks until speech is detected, then records until silence
                record_result = self._recorder.record()
                
                # Handle timeout/abort
                if record_result is None or isinstance(record_result, tuple) and record_result[0] is None:
                    continue
                    
                wav_path, overlapped = record_result if isinstance(record_result, tuple) else (record_result, False)

                self._set_state("PROCESSING")
                result = self._stt.transcribe(wav_path)
                text = result.get("text", "").strip()
                
                # Clean up the temp audio file
                AudioRecorder.cleanup(wav_path)
                
                if not text:
                    self._set_state("READY")
                    continue
                    
                show_transcription(
                    text,
                    language=result.get("language", ""),
                    confidence=result.get("confidence", 0),
                )
                
                # Use context filter to decide if we should respond
                should_respond, clean_text, pre_routed_result = self._context_filter.should_respond(
                    text, self._intent_router, overlapped_with_assistant=overlapped
                )
                
                if should_respond:
                    # Fire off the response handling in a background thread
                    # This allows the main loop to immediately go back to LISTENING
                    # so the user can interrupt S.Y.N. while she is speaking!
                    self._start_response_thread(clean_text, result, pre_routed_result)
                else:
                    self._set_state("READY")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received.")
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)
                time.sleep(1)

    def stop(self):
        """Stop the listen loop."""
        self._interrupted_loop = True
        logger.info("Listen loop stopped.")

    def _start_response_thread(self, text: str, stt_result: dict, pre_routed_result=None):
        # Kill existing response thread if any
        with self._lock:
            if self._current_state == "RESPONDING":
                logger.info("Interruption detected! Halting previous response.")
                self._interrupted_response = True
                stop_speech()
                
        # Brief pause to let the old thread exit cleanly
        time.sleep(0.1)
        
        with self._lock:
            self._interrupted_response = False
            
        threading.Thread(
            target=self._handle_command, 
            args=(text, stt_result, pre_routed_result), 
            daemon=True
        ).start()

    def _handle_command(self, text: str, stt_result: dict, route_result=None):
        """
        Process a transcribed command through intent routing, streaming dispatching,
        and reading back responses sentence-by-sentence.
        """
        self._recorder.set_assistant_speaking(True)
        self._set_state("RESPONDING")

        # Detect user's language and tell TTS to lock voice for this response
        user_lang = stt_result.get("language", "en")
        from backend.voice import tts as tts_module
        tts_module.set_language_hint(user_lang)

        # 1. Route the query if not already routed
        if route_result is None:
            route_result = self._intent_router.route(text)
            
        logger.info(f"Routed: \"{text}\" -> Intent: '{route_result.intent}' (Conf: {route_result.confidence})")

        # 2. Dispatch query and stream speech sentence-by-sentence
        print()
        print("  SYN >> ", end="", flush=True)
        
        chunk_generator = self._dispatcher.dispatch_stream(route_result)
        
        sentence_buffer = ""
        # Hindi uses '।' (Devanagari danda) as a full stop
        sentence_endings = {'.', '?', '!', '।'}
        full_response = []
        
        for chunk in chunk_generator:
            if self._interrupted_response:
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
                
                # Clean up punctuation that breaks TTS
                clean_sentence = sentence.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
                if clean_sentence:
                    enqueue_speech(clean_sentence)
                    
        if not self._interrupted_response:
            # Speak any remaining words in buffer
            leftover = sentence_buffer.strip()
            clean_leftover = leftover.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
            if clean_leftover:
                enqueue_speech(clean_leftover)
                
            print() # Newline at the end
            print()
            
            # Wait for all chunks to finish speaking
            wait_speech_complete()
            
            # Since S.Y.N. successfully finished speaking without interruption,
            # we open the conversation window so the user can follow up!
            self._context_filter.mark_assistant_spoke()
        else:
            print("\n  [Response Interrupted]")
            logger.info("Pipeline was interrupted by user.")
        
        # Reset language hint after response completes
        tts_module.set_language_hint(None)
        
        final_text = "".join(full_response).strip()
        logger.info(f"Pipeline executed successfully. Spoke: \"{final_text}\"")
        self._recorder.set_assistant_speaking(False)
        
        # Only set to READY if we haven't been interrupted by a new LISTENING phase
        with self._lock:
            if not self._interrupted_response:
                self._set_state("READY")
