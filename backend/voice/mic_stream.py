"""
S.Y.N. — Shared Microphone Stream Manager
==========================================
Provides a single managed PyAudio mic stream that multiple modules
(clap detection, audio recording, etc.) can use without conflicts.

Usage:
    from backend.voice.mic_stream import MicStream

    mic = MicStream()
    mic.start()

    # Read raw audio chunks
    chunk = mic.read()

    mic.stop()
"""

import pyaudio
import numpy as np
import threading
import config
from backend.utils.logger import get_logger

logger = get_logger("MIC")


class MicStream:
    """
    Thread-safe microphone stream wrapper.
    Opens one PyAudio stream and lets consumers read chunks from it.
    """

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int | None = None,
        chunk_size: int | None = None,
        channels: int | None = None,
    ):
        # Pull defaults from config, allow overrides
        self.device_index = device_index if device_index is not None else config.MIC_DEVICE_INDEX
        self.sample_rate = sample_rate or config.MIC_SAMPLE_RATE
        self.chunk_size = chunk_size or config.MIC_CHUNK_SIZE
        self.channels = channels or config.MIC_CHANNELS

        self._pyaudio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._lock = threading.Lock()
        self._is_running = False

    # ── Lifecycle ─────────────────────────────

    def start(self):
        """Open the mic stream."""
        with self._lock:
            if self._is_running:
                return

            self._pyaudio = pyaudio.PyAudio()

            # Validate device index
            if self.device_index is not None:
                info = self._pyaudio.get_device_info_by_index(self.device_index)
                if info["maxInputChannels"] < 1:
                    raise ValueError(
                        f"Device {self.device_index} ({info['name']}) has no input channels."
                    )

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
            self._is_running = True

            dev_name = self._get_device_name()
            logger.info(f"Stream opened - device: {dev_name}, "
                        f"rate: {self.sample_rate} Hz, chunk: {self.chunk_size}")

    def stop(self):
        """Close the mic stream and release resources."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            if self._pyaudio is not None:
                self._pyaudio.terminate()
                self._pyaudio = None

            logger.info("Stream closed.")

    # ── Reading ───────────────────────────────

    def read(self) -> np.ndarray:
        """
        Read one chunk of audio data from the mic.
        Returns a numpy int16 array of audio samples.
        Raises RuntimeError if the stream isn't running.
        """
        if not self._is_running or self._stream is None:
            raise RuntimeError("Mic stream is not running. Call start() first.")

        raw_data = self._stream.read(self.chunk_size, exception_on_overflow=False)
        return np.frombuffer(raw_data, dtype=np.int16)

    def read_raw(self) -> bytes:
        """Read one chunk as raw bytes (for saving to WAV, etc.)."""
        if not self._is_running or self._stream is None:
            raise RuntimeError("Mic stream is not running. Call start() first.")

        return self._stream.read(self.chunk_size, exception_on_overflow=False)

    # ── Properties ────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ── Helpers ───────────────────────────────

    def _get_device_name(self) -> str:
        """Get the name of the active input device."""
        if self._pyaudio is None:
            return "unknown"
        if self.device_index is not None:
            info = self._pyaudio.get_device_info_by_index(self.device_index)
        else:
            info = self._pyaudio.get_default_input_device_info()
        return info.get("name", "unknown")

    def list_devices(self) -> list[dict]:
        """List all available audio input devices. Useful for picking MIC_DEVICE_INDEX."""
        pa = self._pyaudio or pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:  # input devices only
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        if self._pyaudio is None:
            pa.terminate()
        return devices

    # ── Context Manager ───────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
