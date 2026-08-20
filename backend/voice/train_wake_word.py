"""
S.Y.N. — Fast Custom Wake Word Trainer for 'Hey SYN' / 'Sin'
============================================================
Trains a custom neural wake-word detector specifically tuned to "SYN" / "Sin".
Uses local TTS and fast Edge-TTS batching, extracting openWakeWord embeddings.
"""

import os
import tempfile
import asyncio
import miniaudio
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from openwakeword.utils import AudioFeatures

# Target phonetic variations
POSITIVE_PHRASES = [
    "Hey Sin", "Sin", "Hey SYN", "SYN",
    "Hi Sin", "Hi SYN", "Okay Sin", "Okay SYN",
    "Wake up Sin", "Wake up SYN", "Hey there Sin",
    "Hey Syn", "Syn", "Okay Syn"
]

NEGATIVE_PHRASES = [
    "Hey Jarvis", "Jarvis", "Alexa", "Hey Siri", "Hey Google",
    "Computer", "Hey Mycroft", "Hello everyone", "What time is it",
    "How is the weather today", "Open Google Chrome", "Open Visual Studio",
    "Turn up the volume", "Play some music", "Stop talking", "Close window",
    "Yes", "No", "Thanks", "Good morning", "Tell me a joke",
    "Can you hear me", "This is a test", "Never mind", "Cancel that",
    "Synthesize", "Synthesis", "Sync", "Synchronize", "Synergy",
    "Simple", "Single", "Signal", "System", "Python", "Code"
]

VOICES = [
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-DavisNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"
]


def decode_and_resample(mp3_path: str, target_sr: int = 16000) -> np.ndarray:
    """Decodes MP3 to 16kHz mono int16 numpy array."""
    f = miniaudio.decode_file(mp3_path)
    samples = np.frombuffer(f.samples, dtype=np.int16)
    if f.nchannels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if f.sample_rate != target_sr:
        num_target_samples = int(len(samples) * target_sr / f.sample_rate)
        samples = scipy.signal.resample(samples.astype(np.float32), num_target_samples).astype(np.int16)
    return samples


def extract_features_from_audio(audio_data: np.ndarray, af: AudioFeatures) -> list[np.ndarray]:
    """Feeds audio through openWakeWord AudioFeatures and extracts 1536-dim feature windows."""
    af.reset()
    features = []
    
    # Pad audio with 0.5s silence at beginning and end
    silence = np.zeros(8000, dtype=np.int16)
    full_audio = np.concatenate([silence, audio_data, silence])
    
    # Stream in 1280 sample chunks (80ms)
    chunk_size = 1280
    for i in range(0, len(full_audio) - chunk_size, chunk_size):
        chunk = full_audio[i:i + chunk_size]
        af(chunk)
        feat = af.get_features()
        if feat is not None and feat.shape == (1, 16, 96):
            features.append(feat[0].flatten())
            
    return features


async def generate_edge_clip(text: str, voice: str, rate: str, output_path: str):
    import edge_tts
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        return output_path
    except Exception:
        return None


async def build_dataset():
    temp_dir = os.path.join(tempfile.gettempdir(), "syn_wakeword_fast")
    os.makedirs(temp_dir, exist_ok=True)
    
    af = AudioFeatures()
    X = []
    y = []
    
    print("Generating positive training samples for 'SYN' / 'Sin'...")
    tasks = []
    meta = []
    
    count = 0
    for phrase in POSITIVE_PHRASES:
        for voice in VOICES:
            for rate in ["-10%", "+0%", "+10%"]:
                mp3_path = os.path.join(temp_dir, f"pos_{count}.mp3")
                tasks.append(generate_edge_clip(phrase, voice, rate, mp3_path))
                meta.append((mp3_path, 1))
                count += 1
                
    for phrase in NEGATIVE_PHRASES:
        for voice in VOICES[:4]:
            mp3_path = os.path.join(temp_dir, f"neg_{count}.mp3")
            tasks.append(generate_edge_clip(phrase, voice, "+0%", mp3_path))
            meta.append((mp3_path, 0))
            count += 1
            
    # Run TTS generation in parallel batches of 25
    batch_size = 25
    print(f"Executing {len(tasks)} TTS generation tasks in parallel batches...")
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])
        
    print("Extracting acoustic embedding features via ONNX...")
    for path, label in meta:
        if os.path.exists(path):
            try:
                audio = decode_and_resample(path)
                feats = extract_features_from_audio(audio, af)
                if feats:
                    if label == 1:
                        # For positive, use the last 3-4 feature windows containing the utterance
                        for f in feats[-4:]:
                            X.append(f)
                            y.append(1)
                    else:
                        for f in feats:
                            X.append(f)
                            y.append(0)
            except Exception:
                pass
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
                    
    # Add random background noise as negatives
    for _ in range(40):
        noise = (np.random.randn(32000) * 150).astype(np.int16)
        feats = extract_features_from_audio(noise, af)
        for f in feats[:3]:
            X.append(f)
            y.append(0)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset compiled: {len(X)} total vectors (Pos: {np.sum(y == 1)}, Neg: {np.sum(y == 0)})")
    return X, y


def train_and_save(X, y):
    print("Training neural classifier (MLP)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=300,
            random_state=42,
            early_stopping=True
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'SYN / Sin']))
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "hey_syn.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n[SUCCESS] Saved custom wake word model to: {model_path}")
    return model_path


if __name__ == "__main__":
    X, y = asyncio.run(build_dataset())
    train_and_save(X, y)
