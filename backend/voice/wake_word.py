"""
S.Y.N. — Wake Word Engine (openWakeWord + Custom 'Hey SYN' Model)
================================================================
Lightweight, local, 100% offline wake word detection.
Supports both custom trained models ('hey_syn' / 'sin') and openWakeWord defaults.
"""

import os
import time
import numpy as np
import config
import joblib
from backend.utils.logger import get_logger
from backend.voice.mic_stream import MicStream
from openwakeword.utils import AudioFeatures

logger = get_logger("WAKEWORD")


class WakeWordDetector:
    """
    Listens for configured wake words using openWakeWord neural feature extraction
    and custom 'Hey SYN' / 'Sin' classification.
    """

    def __init__(self, model_name: str | None = None, threshold: float = 0.5):
        self.model_name = (model_name or getattr(config, "WAKE_WORD_MODEL", "hey_syn")).lower()
        self.threshold = getattr(config, "WAKE_WORD_THRESHOLD", threshold)
        self._is_custom = False
        self._custom_model = None
        self._af = None
        self._oww_model = None
        self._mic = None
        self._load_model()

    def _load_model(self):
        """Loads custom 'hey_syn' model or default openWakeWord model."""
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        custom_joblib = os.path.join(models_dir, "hey_syn.joblib")

        if self.model_name in ["hey_syn", "syn", "sin", "hey_sin"] and os.path.exists(custom_joblib):
            logger.info(f"Loading custom SYN / Sin neural wake-word model from {custom_joblib}...")
            self._is_custom = True
            self._custom_model = joblib.load(custom_joblib)
            self._af = AudioFeatures()
            logger.info(f"Custom wake word 'SYN' / 'Sin' model ready (threshold: {self.threshold}).")
        else:
            try:
                from openwakeword.model import Model
                logger.info(f"Loading standard openWakeWord model: '{self.model_name}'...")
                self._is_custom = False
                self._oww_model = Model(wakeword_models=[self.model_name])
                logger.info(f"Wake word model '{self.model_name}' ready.")
            except Exception as e:
                logger.error(f"Failed to load wake word model: {e}", exc_info=True)
                raise e

    def listen_for_wake_word(self, existing_mic: MicStream = None, timeout: float | None = None) -> bool:
        """
        Blocks until the wake word is spoken or timeout is reached.
        """
        mic = existing_mic or MicStream(sample_rate=16000, chunk_size=1280)
        owns_mic = existing_mic is None

        if not mic.is_running:
            mic.start()

        start_time = time.time()

        if self._is_custom:
            self._af.reset()
        else:
            self._oww_model.reset()

        logger.debug(f"Listening for wake word '{self.model_name}'...")

        try:
            while mic.is_running:
                if timeout and (time.time() - start_time) > timeout:
                    return False

                # Read 1280 samples (80ms at 16kHz)
                raw_chunk = mic.read_raw()
                audio_chunk = np.frombuffer(raw_chunk, dtype=np.int16)

                if self._is_custom:
                    self._af(audio_chunk)
                    feat = self._af.get_features()
                    if feat is not None and feat.shape == (1, 16, 96):
                        flat_feat = feat[0].reshape(1, -1)
                        probs = self._custom_model.predict_proba(flat_feat)
                        score = float(probs[0][1])
                        if score >= self.threshold:
                            logger.info(f"Wake word 'SYN' (Sin) DETECTED! (confidence: {score:.3f})")
                            self._af.reset()
                            return True
                else:
                    prediction = self._oww_model.predict(audio_chunk)
                    score = prediction.get(self.model_name, 0.0)
                    if score >= self.threshold:
                        logger.info(f"Wake word '{self.model_name}' DETECTED! (confidence: {score:.3f})")
                        self._oww_model.reset()
                        return True

        except Exception as e:
            logger.error(f"Error in wake word detection loop: {e}")
            return False
        finally:
            if owns_mic:
                mic.stop()

        return False
