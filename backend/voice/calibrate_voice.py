"""
S.Y.N. — User Voice Calibration & Custom Wake Word Trainer
==========================================================
Records your actual voice saying "Hey SYN" / "SYN",
extracts your unique vocal embeddings, and trains a personalized
wake-word model tailored specifically to your microphone and voice.

Usage:
    python backend/voice/calibrate_voice.py
"""

import os
import time
import miniaudio
import numpy as np
import scipy.io.wavfile as wavfile
import config
from backend.voice.mic_stream import MicStream
from openwakeword.utils import AudioFeatures
from backend.voice.train_wake_word import (
    build_dataset,
    extract_features_from_audio,
    train_and_save
)


def record_sample(sample_num: int, total_samples: int, phrase: str, duration: float = 2.5) -> np.ndarray:
    """Records a single audio sample of the user speaking the wake word."""
    mic = MicStream(sample_rate=16000, chunk_size=1280, channels=1)
    mic.start()

    print("\n" + "=" * 55)
    print(f"  [Sample {sample_num}/{total_samples}] Target Phrase: \"{phrase}\"")
    print(f"  Get ready... speaking in:")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(0.7)

    print(f"  >>> SPEAK NOW: \"{phrase}\" <<<")
    frames = []
    start = time.time()

    while (time.time() - start) < duration:
        raw_chunk = mic.read_raw()
        frames.append(raw_chunk)

    mic.stop()
    print("  [Recorded! Processing sample...]")

    audio_bytes = b"".join(frames)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_data


def calibrate_and_train():
    print("=" * 60)
    print("  🎙️ S.Y.N. Voice Personalization & Wake Word Calibration")
    print("  We will record 5 quick clips of your voice saying 'Hey SYN'")
    print("=" * 60)

    phrases = [
        "Hey SYN",
        "SYN",
        "Hey SYN",
        "SYN",
        "Hey SYN"
    ]

    user_samples = []
    for idx, phrase in enumerate(phrases, 1):
        audio = record_sample(idx, len(phrases), phrase)
        user_samples.append(audio)
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("  Analyzing your voice and extracting neural embeddings...")
    print("=" * 60)

    af = AudioFeatures()
    user_features = []

    for idx, audio in enumerate(user_samples):
        feats = extract_features_from_audio(audio, af)
        if feats:
            # Replicate user voice features to give strong weight to user's exact voice
            for f in feats[-5:]:
                for _ in range(15):  # 15x weight for user's real voice
                    user_features.append(f)

    print(f"Extracted {len(user_features)} personalized acoustic vectors from your voice.")

    # Now load synthetic base dataset and merge with user voice
    import asyncio
    print("\nCompiling full dataset with your voice as primary target...")
    X_base, y_base = asyncio.run(build_dataset())

    # Add user voice as positive label (1)
    X_user = np.array(user_features)
    y_user = np.ones(len(user_features), dtype=int)

    X_combined = np.vstack([X_base, X_user])
    y_combined = np.concatenate([y_base, y_user])

    print(f"\nFinal training set size: {len(X_combined)} samples (User-weighted Positives: {np.sum(y_combined == 1)})")

    # Train and save the personalized model
    model_path = train_and_save(X_combined, y_combined)

    print("\n" + "=" * 60)
    print("  🎉 CALIBRATION COMPLETE!")
    print(f"  Your personalized wake word model is ready: {model_path}")
    print("  You can now run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    calibrate_and_train()
