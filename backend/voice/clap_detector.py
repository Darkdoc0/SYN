"""
S.Y.N. — Clap Detection Engine
================================
Detects double-clap (or single-clap) patterns from a live mic stream
and triggers a callback to wake SYN.

How it works:
    1. Continuously reads audio chunks from MicStream
    2. Calculates RMS energy of each chunk
    3. A "spike" above the threshold = potential clap
    4. Validates the spike is short (clap-like, not speech/music)
    5. Waits for a second clap within the time window (double-clap mode)
    6. On confirmed pattern → fires the on_wake callback
    7. Enters cooldown to prevent re-triggers

Usage:
    from backend.voice.clap_detector import ClapDetector

    def my_wake_handler():
        print("SYN woke up!")

    detector = ClapDetector(on_wake=my_wake_handler)
    detector.start()  # blocks in listen loop (run in thread if needed)
"""

import time
import threading
import numpy as np
from backend.voice.mic_stream import MicStream
import config
from backend.utils.logger import get_logger

logger = get_logger("CLAP")


class ClapDetector:
    """
    Listens for clap patterns on the mic and calls on_wake when detected.
    """

    def __init__(self, on_wake=None):
        """
        Args:
            on_wake: Callback function — called when clap pattern is detected.
                     Receives no arguments.
        """
        self.on_wake = on_wake

        # Config
        self.energy_threshold = config.CLAP_ENERGY_THRESHOLD
        self.double_tap_window = config.CLAP_DOUBLE_TAP_WINDOW
        self.cooldown = config.CLAP_COOLDOWN
        self.pattern = config.CLAP_PATTERN  # "single" or "double"
        self.min_duration_ms = config.CLAP_MIN_DURATION_MS
        self.max_duration_ms = config.CLAP_MAX_DURATION_MS

        # State
        self._mic = MicStream()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_trigger_time = 0.0
        self._first_clap_time: float | None = None
        self._assistant_speaking = False

        # Rolling energy for adaptive baseline (background noise tracking)
        self._energy_history: list[float] = []
        self._energy_history_max = 50  # number of chunks to track

    # ── Public API ────────────────────────────

    def start(self, blocking: bool = True):
        """
        Start listening for claps.

        Args:
            blocking: If True, blocks the current thread.
                      If False, runs in a background thread.
        """
        self._running = True
        self._mic.start()

        logger.info(f"Detector started - mode: {self.pattern}, "
                    f"threshold: {self.energy_threshold}")

        if blocking:
            self._listen_loop()
        else:
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop listening."""
        self._running = False
        self._mic.stop()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Detector stopped.")

    def set_threshold(self, value: int):
        """Adjust clap energy threshold at runtime."""
        self.energy_threshold = value
        logger.debug(f"Threshold updated to {value}")
        
    def set_assistant_speaking(self, is_speaking: bool):
        """Called by the listen loop when S.Y.N. starts/stops speaking."""
        self._assistant_speaking = is_speaking
        if is_speaking:
            # Artificially inflate the background history so her first few words 
            # don't accidentally trigger a false clap before the rolling average adapts!
            inflated = self.energy_threshold * 2.5
            self._energy_history = [inflated] * self._energy_history_max
            logger.debug(f"Assistant speaking: Inflated background noise to {inflated:.0f}")

    # ── Core Loop ─────────────────────────────

    def _listen_loop(self):
        """
        Main detection loop. Reads audio chunks and watches for clap patterns.
        """
        # Track consecutive above-threshold chunks for spike duration
        spike_start: float | None = None

        while self._running:
            try:
                audio_chunk = self._mic.read()
                energy = self._calculate_rms(audio_chunk)
                current_time = time.time()

                # We will update energy history *after* checking for spikes,
                # so that massive clap spikes don't inflate the background baseline.

                # ── Check cooldown ──
                if (current_time - self._last_trigger_time) < self.cooldown:
                    continue

                # ── Detect energy spike ──
                # Adaptive threshold: 2.5x the rolling background noise (12.0x if speaking!)
                current_bg = self.get_background_noise_level()
                multiplier = 12.0 if self._assistant_speaking else 2.5
                dynamic_threshold = max(self.energy_threshold, current_bg * multiplier)
                
                is_spike = energy > dynamic_threshold

                if is_spike:
                    if spike_start is None:
                        spike_start = current_time
                else:
                    self._update_energy_history(energy)
                    
                    # Spike ended — validate duration
                    if spike_start is not None:
                        spike_duration_ms = (current_time - spike_start) * 1000

                        if self.min_duration_ms <= spike_duration_ms <= self.max_duration_ms:
                            # Valid clap detected!
                            self._handle_clap(current_time)

                            logger.info(f"*CLAP* detected! energy={energy:.0f}, "
                                        f"duration={spike_duration_ms:.1f}ms")

                        spike_start = None

                # ── Double-clap timeout (reset if too slow) ──
                if (self._first_clap_time is not None and
                        (current_time - self._first_clap_time) > self.double_tap_window):
                    self._first_clap_time = None
                    logger.debug("First clap timed out - reset.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(0.1)

        self._mic.stop()

    def _handle_clap(self, current_time: float):
        """
        Process a validated clap event. Handles single vs double pattern.
        """
        if self.pattern == "single":
            # Single clap mode — trigger immediately
            self._trigger_wake()
            self._last_trigger_time = current_time

        elif self.pattern == "double":
            if self._first_clap_time is None:
                # First clap — wait for second
                self._first_clap_time = current_time
                logger.debug("First clap registered - waiting for second...")
            else:
                # Second clap within window — TRIGGER!
                time_between = current_time - self._first_clap_time
                if time_between <= self.double_tap_window:
                    self._trigger_wake()
                    self._last_trigger_time = current_time
                self._first_clap_time = None

    def _trigger_wake(self):
        """Fire the wake callback."""
        logger.info(">> WAKE TRIGGERED! <<")

        if self.on_wake is not None:
            # Run callback in a separate thread to not block detection
            threading.Thread(target=self.on_wake, daemon=True).start()

    # ── Audio Analysis ────────────────────────

    @staticmethod
    def _calculate_rms(audio_chunk: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) energy of an audio chunk.
        Higher RMS = louder sound.
        """
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))

    def _update_energy_history(self, energy: float):
        """Track rolling energy for background noise awareness."""
        self._energy_history.append(energy)
        if len(self._energy_history) > self._energy_history_max:
            self._energy_history.pop(0)

    def get_background_noise_level(self) -> float:
        """Return the average background noise RMS (useful for calibration)."""
        if not self._energy_history:
            return 0.0
        return float(np.mean(self._energy_history))

    # ── Context Manager ───────────────────────

    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, *args):
        self.stop()


# ──────────────────────────────────────────────
#  Calibration Helper
# ──────────────────────────────────────────────
def calibrate_clap_threshold(duration: float = 5.0) -> dict:
    """
    Listen to ambient noise for `duration` seconds, then report
    suggested threshold values. Run this to tune CLAP_ENERGY_THRESHOLD
    for your room.

    Returns dict with:
        - avg_noise: average background RMS
        - max_noise: peak background RMS
        - suggested_threshold: recommended CLAP_ENERGY_THRESHOLD
    """
    print(f"\n[MIC] Calibrating... stay quiet for {duration} seconds.\n")

    mic = MicStream()
    mic.start()

    energies = []
    end_time = time.time() + duration

    while time.time() < end_time:
        chunk = mic.read()
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        energies.append(rms)

    mic.stop()

    avg_noise = float(np.mean(energies))
    max_noise = float(np.max(energies))
    # Suggested threshold: 3x above the max background noise
    suggested = max(int(max_noise * 3), 1000)

    print(f"[RESULTS]")
    print(f"   Average noise : {avg_noise:.0f}")
    print(f"   Peak noise    : {max_noise:.0f}")
    print(f"   Suggested threshold: {suggested}")
    print(f"\n   >> Set CLAP_ENERGY_THRESHOLD = {suggested} in config.py\n")

    return {
        "avg_noise": avg_noise,
        "max_noise": max_noise,
        "suggested_threshold": suggested,
    }


# ──────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibrate_clap_threshold()
    else:
        print("=" * 50)
        print("  S.Y.N. — Clap Detector Test")
        print("  Double-clap to trigger wake!")
        print("  Ctrl+C to exit")
        print("=" * 50)

        def on_wake():
            print("\n>> SYN WOKE UP! (would start listening here) <<\n")

        detector = ClapDetector(on_wake=on_wake)

        try:
            detector.start(blocking=True)
        except KeyboardInterrupt:
            detector.stop()
            print("\nDetector stopped.")
