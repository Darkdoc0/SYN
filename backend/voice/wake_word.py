"""
S.Y.N. — Wake Word Engine (openWakeWord)
========================================
Lightweight, local, 100% offline wake word detection powered by openWakeWord.
Listens continuously at near-zero CPU and triggers the main listen loop.
"""

import time
import numpy as np
import config
from backend.utils.logger import get_logger
from backend.voice.mic_stream import MicStream

logger = get_logger("WAKEWORD")


class WakeWordDetector:
    """
    Listens for configured wake words using openWakeWord neural network models.
    """

    def __init__(self, model_name: str | None = None, threshold: float = 0.5):
        self.model_name = model_name or getattr(config, "WAKE_WORD_MODEL", "hey_jarvis")
        self.threshold = getattr(config, "WAKE_WORD_THRESHOLD", threshold)
        self._model = None
        self._mic = None
        self._load_model()

    def _load_model(self):
        """Loads the openWakeWord ONNX model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            logger.info(f"Loading openWakeWord model: '{self.model_name}' (threshold: {self.threshold})...")
            self._model = Model(wakeword_models=[self.model_name])
            logger.info(f"Wake word model '{self.model_name}' ready.")
        except Exception as e:
            logger.error(f"Failed to load openWakeWord model: {e}", exc_info=True)
            raise e

    def listen_for_wake_word(self, existing_mic: MicStream = None, timeout: float | None = None) -> bool:
        """
        Blocks until the wake word is spoken or timeout is reached.
        
        Args:
            existing_mic: Optional MicStream instance.
            timeout: Optional max seconds to wait.
            
        Returns:
            True if wake word detected, False if timeout or stopped.
        """
        mic = existing_mic or MicStream(sample_rate=16000, chunk_size=1280)
        owns_mic = existing_mic is None

        if not mic.is_running:
            mic.start()

        start_time = time.time()
        self._model.reset()

        logger.debug(f"Listening for wake word '{self.model_name}'...")

        try:
            while mic.is_running:
                if timeout and (time.time() - start_time) > timeout:
                    return False

                # Read 1280 samples (80ms at 16kHz)
                raw_chunk = mic.read_raw()
                audio_chunk = np.frombuffer(raw_chunk, dtype=np.int16)

                # Feed to openWakeWord
                prediction = self._model.predict(audio_chunk)

                # Check prediction score
                score = prediction.get(self.model_name, 0.0)
                if score >= self.threshold:
                    logger.info(f"Wake word '{self.model_name}' DETECTED! (confidence: {score:.3f})")
                    self._model.reset()
                    return True

        except Exception as e:
            logger.error(f"Error in wake word detection loop: {e}")
            return False
        finally:
            if owns_mic:
                mic.stop()

        return False
