"""
S.Y.N. — Audio Recorder
=========================
Records audio from the microphone until the user stops speaking.

HOW IT WORKS (learn this!):
──────────────────────────
Audio comes in as a stream of tiny "chunks" — think of it like a conveyor
belt of sound. Each chunk is ~23ms of audio (1024 samples at 44100 Hz).

For each chunk, we calculate its "energy" (loudness) using RMS:
    RMS = sqrt(mean(samples²))

The recorder has 3 states:
    1. WAITING  — Mic is open, but user hasn't spoken yet.
                  We skip silence at the start so the recording
                  doesn't begin with dead air.
    2. RECORDING — User is speaking! We save every chunk.
    3. SILENCE   — User went quiet. If silence lasts longer than
                   SILENCE_TIMEOUT (2 sec), we stop recording.

The final audio is saved as a WAV file, which Whisper can read.

WAV FILE FORMAT (good to know!):
────────────────────────────────
WAV is the simplest audio format — just raw samples with a header.
Header says: "I have 1 channel, 44100 samples/sec, 16 bits per sample"
Then it's just the raw numbers. No compression, no codec.
That's why Whisper loves it — no decoding needed.
"""

import wave
import time
import tempfile
import os
import numpy as np
from backend.voice.mic_stream import MicStream
import config
from backend.utils.logger import get_logger

logger = get_logger("REC")


class AudioRecorder:
    """
    Records audio from the mic with automatic silence detection.
    
    Usage:
        recorder = AudioRecorder()
        wav_path = recorder.record()  # blocks until speech + silence
        # wav_path is now a file like "C:/Users/.../syn_audio_abc123.wav"
    """

    def __init__(self):
        # ── Thresholds ──
        # Speech threshold: audio louder than this = "someone is talking"
        # We use a lower threshold than clap detection because speech
        # is quieter than a clap but still clearly above silence.
        self.speech_threshold = config.CLAP_ENERGY_THRESHOLD * 0.4

        # How long to wait for the user to START speaking (seconds)
        self.max_wait_for_speech = 8.0

        # How long silence must last to STOP recording (seconds)
        self.silence_timeout = config.STT_SILENCE_TIMEOUT  # 2.0s

        # Maximum recording length to prevent infinite recordings
        self.max_record_time = config.STT_MAX_RECORD_TIME  # 15.0s

        # The mic stream (shared with clap detector)
        self._mic = MicStream()

    def record(self, existing_mic: MicStream = None) -> str | None:
        """
        Record audio until the user stops speaking.

        Args:
            existing_mic: If provided, uses this mic stream instead of
                         creating a new one. Useful when the clap detector
                         already has the mic open.

        Returns:
            Path to the recorded WAV file, or None if no speech detected.

        WHAT HAPPENS STEP BY STEP:
            1. Open mic (or reuse existing one)
            2. Wait for the user to start speaking (skip initial silence)
            3. Record while they're speaking
            4. When silence lasts > 2 seconds, stop
            5. Save to a temporary .wav file
            6. Return the file path
        """
        # Use existing mic or create new one
        mic = existing_mic or self._mic
        owns_mic = existing_mic is None  # we only close it if we opened it

        if not mic.is_running:
            mic.start()

        logger.info(f"Listening for speech... (threshold: {self.speech_threshold:.0f})")

        # ── State tracking ──
        frames: list[bytes] = []       # collected audio chunks (raw bytes)
        speech_detected = False        # has the user started talking?
        silence_start: float = None    # when did the current silence begin?
        record_start: float = None     # when did we start saving audio?
        wait_start = time.time()       # when did we start waiting?

        # ── Main recording loop ──
        while True:
            # Read one chunk from the mic
            # WHY read_raw()? Because we need the raw bytes to save to WAV.
            # The numpy array (from read()) is for analysis, raw bytes are for saving.
            raw_chunk = mic.read_raw()
            audio_chunk = np.frombuffer(raw_chunk, dtype=np.int16)

            # Calculate energy (loudness) of this chunk
            energy = self._calculate_rms(audio_chunk)
            is_speech = energy > self.speech_threshold

            # ── STATE 1: Waiting for speech to begin ──
            if not speech_detected:
                if is_speech:
                    # User started talking!
                    speech_detected = True
                    record_start = time.time()
                    frames.append(raw_chunk)  # don't lose this first chunk

                    logger.info(f"Speech detected! (energy: {energy:.0f}) Recording...")

                elif (time.time() - wait_start) > self.max_wait_for_speech:
                    # User didn't say anything — timeout
                    logger.info("No speech detected. Timeout.")
                    if owns_mic:
                        mic.stop()
                    return None

                continue  # keep waiting for speech

            # ── STATE 2 & 3: Recording (speech or silence) ──
            frames.append(raw_chunk)

            if is_speech:
                # User is still talking — reset silence timer
                silence_start = None
            else:
                # Silence detected
                if silence_start is None:
                    silence_start = time.time()

                # Check if silence has lasted long enough to stop
                # WHY 2 seconds? Natural speech has short pauses (0.3-0.5s).
                # We need to wait longer than that to avoid cutting off
                # mid-sentence. 2 seconds is a good balance.
                silence_duration = time.time() - silence_start
                if silence_duration >= self.silence_timeout:
                    logger.info(f"Silence for {silence_duration:.1f}s. Stopping.")
                    break

            # Safety: max recording time
            if (time.time() - record_start) >= self.max_record_time:
                logger.info(f"Max recording time ({self.max_record_time}s) reached.")
                break

        # ── Save to WAV file ──
        if owns_mic:
            mic.stop()

        if not frames:
            return None

        wav_path = self._save_wav(frames, mic.sample_rate, mic.channels)

        duration = len(frames) * mic.chunk_size / mic.sample_rate
        logger.info(f"Saved {duration:.1f}s of audio to: {wav_path}")

        return wav_path

    def _save_wav(self, frames: list[bytes], sample_rate: int, channels: int) -> str:
        """
        Save raw audio frames to a WAV file.

        WHY A TEMP FILE?
            We save to a temporary file because we only need it long enough
            for Whisper to read it. After transcription, it can be deleted.
            tempfile.mktemp() gives us a unique filename in the OS temp dir.

        WAV STRUCTURE:
            - Header: RIFF marker, file size, format info
            - Data: raw PCM samples (just numbers representing air pressure)
            Python's `wave` module handles the header for us.
        """
        # Create a temp file path (we manage deletion ourselves)
        wav_path = os.path.join(
            tempfile.gettempdir(),
            f"syn_audio_{int(time.time())}.wav"
        )

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(channels)           # 1 = mono
            wf.setsampwidth(config.MIC_FORMAT_WIDTH)  # 2 = 16-bit
            wf.setframerate(sample_rate)         # 44100 Hz
            wf.writeframes(b"".join(frames))     # all audio data

        return wav_path

    @staticmethod
    def _calculate_rms(audio_chunk: np.ndarray) -> float:
        """
        Root Mean Square — measures the "average loudness" of audio.

        MATH BREAKDOWN:
            1. Square every sample:     [-100, 50, -80]  →  [10000, 2500, 6400]
            2. Take the mean:           (10000 + 2500 + 6400) / 3 = 6300
            3. Square root:             sqrt(6300) ≈ 79.4

        Why square first? Because audio samples are positive AND negative
        (sound waves oscillate). Squaring makes everything positive,
        then we average and square-root to get back to the original scale.
        """
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))

    @staticmethod
    def cleanup(wav_path: str):
        """Delete a temporary WAV file after we're done with it."""
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass


# ──────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  S.Y.N. -- Audio Recorder Test")
    print("  Speak after the prompt, then go silent.")
    print("=" * 50)
    print()

    recorder = AudioRecorder()

    print("Say something...")
    wav_file = recorder.record()

    if wav_file:
        # Get file size to verify it recorded something
        size_kb = os.path.getsize(wav_file) / 1024
        print(f"\nRecording saved: {wav_file}")
        print(f"File size: {size_kb:.1f} KB")
        print("(This file will be fed to Whisper for transcription)")

        # Don't delete — let the user inspect the file
        print(f"\nYou can play it at: {wav_file}")
    else:
        print("\nNo speech detected.")
