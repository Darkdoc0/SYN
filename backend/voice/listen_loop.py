"""
S.Y.N. — Listen Loop (JARVIS Mode with openWakeWord)
===================================================
Orchestrates wake word detection, audio recording, Whisper STT,
intent dispatching, and conversational follow-up windows.
"""

import threading
import time
from backend.voice.tts import speak, stop_speech, enqueue_speech, wait_speech_complete
from backend.voice.mic_stream import MicStream
from backend.voice.audio_recorder import AudioRecorder
from backend.voice.wake_word import WakeWordDetector
from backend.voice.stt import SpeechToText
from backend.utils.logger import get_logger
from backend.utils.status import show_state, show_transcription
from backend.brain.intent_router import IntentRouter
from backend.brain.dispatcher import Dispatcher
import config

logger = get_logger("LOOP")


class ListenLoop:
    """
    Main SYN orchestrator — Combines openWakeWord detection, Whisper STT,
    and conversational context windows.
    """

    def __init__(self):
        self._mic = MicStream(
            sample_rate=config.MIC_SAMPLE_RATE,
            chunk_size=config.MIC_CHUNK_SIZE,
            channels=config.MIC_CHANNELS,
        )
        self._wake_detector = WakeWordDetector()
        self._recorder = AudioRecorder()
        self._stt = SpeechToText()
        self._intent_router = IntentRouter()
        self._dispatcher = Dispatcher()

        self._interrupted_loop = False
        self._interrupted_response = False
        self._current_state = "READY"
        self._lock = threading.Lock()
        self._last_assistant_speech_time = 0.0

    def _set_state(self, state: str):
        self._current_state = state
        show_state(state)

    def start(self):
        """Start the main wake-word and listen loop (blocking)."""
        logger.info(f"Listen loop starting with wake word: '{config.WAKE_WORD_MODEL}'...")
        self._set_state("READY")
        self._interrupted_loop = False

        # Open shared mic stream
        self._mic.start()

        # Greet on boot
        speak(config.SYN_BOOT_MESSAGE)

        while not self._interrupted_loop:
            try:
                # Check if we are inside the active conversation window
                time_since_speech = time.time() - self._last_assistant_speech_time
                in_conversation_window = (self._last_assistant_speech_time > 0) and (
                    time_since_speech <= config.CONVERSATION_WINDOW
                )

                if in_conversation_window:
                    # Inside conversation window: listen for follow-up speech directly
                    remaining_window = max(1.0, config.CONVERSATION_WINDOW - time_since_speech)
                    logger.info(f"Conversation window active ({remaining_window:.1f}s remaining). Listening for follow-up...")
                    self._set_state("LISTENING")

                    wav_path, overlapped = self._recorder.record(
                        existing_mic=self._mic,
                        max_wait=remaining_window,
                    )

                    if not wav_path:
                        # Conversation window timed out with no follow-up speech
                        logger.info("Conversation window closed. Returning to wake word listening.")
                        self._last_assistant_speech_time = 0.0
                        self._set_state("READY")
                        continue
                else:
                    # Idle mode: sleep and listen for the Wake Word using openWakeWord (0% CPU)
                    self._set_state("READY")
                    detected = self._wake_detector.listen_for_wake_word(existing_mic=self._mic)

                    if not detected:
                        continue

                    # Wake word was detected!
                    logger.info("Wake word detected! Listening for command...")
                    self._set_state("LISTENING")

                    # Record the command
                    wav_path, overlapped = self._recorder.record(
                        existing_mic=self._mic,
                        max_wait=6.0,
                    )

                    if not wav_path:
                        # User woke S.Y.N. up but didn't say a command within 6s
                        logger.info("No command spoken after wake word. Returning to sleep.")
                        self._set_state("READY")
                        continue

                # Process the recorded command audio
                self._set_state("PROCESSING")
                result = self._stt.transcribe(wav_path)
                text = result.get("text", "").strip()

                # Clean up temporary WAV
                AudioRecorder.cleanup(wav_path)

                if not text:
                    self._set_state("READY")
                    continue

                show_transcription(
                    text,
                    language=result.get("language", ""),
                    confidence=result.get("confidence", 0),
                )

                # Execute command and speak response completely before returning to listen
                self._handle_command(text, result)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received.")
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)
                time.sleep(1)

        self._mic.stop()

    def stop(self):
        """Stop the listen loop."""
        self._interrupted_loop = True
        self._mic.stop()
        logger.info("Listen loop stopped.")

    def _handle_command(self, text: str, stt_result: dict):
        """
        Process transcribed command through intent routing, streaming dispatching,
        and TTS playback.
        """
        self._recorder.set_assistant_speaking(True)
        self._set_state("RESPONDING")

        user_lang = stt_result.get("language", "en")
        from backend.voice import tts as tts_module
        tts_module.set_language_hint(user_lang)

        # Route query
        route_result = self._intent_router.route(text)
        logger.info(f"Routed: \"{text}\" -> Intent: '{route_result.intent}' (Conf: {route_result.confidence})")

        # Stream response
        print()
        print("  SYN >> ", end="", flush=True)

        chunk_generator = self._dispatcher.dispatch_stream(route_result)
        sentence_buffer = ""
        sentence_endings = {'.', '?', '!', '।'}
        full_response = []

        for chunk in chunk_generator:
            if self._interrupted_response:
                break

            print(chunk, end="", flush=True)
            full_response.append(chunk)
            sentence_buffer += chunk

            while True:
                end_idx = -1
                ending_len = 0
                for char in sentence_endings:
                    idx = sentence_buffer.find(char + " ")
                    if idx != -1:
                        if end_idx == -1 or idx < end_idx:
                            end_idx = idx
                            ending_len = 2
                    idx = sentence_buffer.find(char + "\n")
                    if idx != -1:
                        if end_idx == -1 or idx < end_idx:
                            end_idx = idx
                            ending_len = 2

                if end_idx == -1:
                    idx = sentence_buffer.find("\n")
                    if idx != -1:
                        end_idx = idx
                        ending_len = 1

                if end_idx == -1:
                    break

                sentence = sentence_buffer[:end_idx + 1].strip()
                sentence_buffer = sentence_buffer[end_idx + ending_len:]
                clean_sentence = sentence.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
                if clean_sentence:
                    enqueue_speech(clean_sentence)

        if not self._interrupted_response:
            leftover = sentence_buffer.strip()
            clean_leftover = leftover.replace('"', '').replace('*', '').replace('_', '').replace('`', '').strip()
            if clean_leftover:
                enqueue_speech(clean_leftover)

            print()
            print()
            wait_speech_complete()

            # Record completion time to open conversation window
            self._last_assistant_speech_time = time.time()
        else:
            print("\n  [Response Interrupted]")
            logger.info("Pipeline was interrupted by user.")

        tts_module.set_language_hint(None)
        final_text = "".join(full_response).strip()
        logger.info(f"Pipeline executed successfully. Spoke: \"{final_text}\"")

        self._recorder.set_assistant_speaking(False)
        self._set_state("READY")
