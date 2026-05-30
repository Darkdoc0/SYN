"""
S.Y.N. — Main Entry Point
===========================
Boots SYN and starts the full pipeline:
    Clap → Wake → Record → Transcribe → (Day 5+: Intent → Action)

RUN WITH:
    python main.py              → Full boot (clap wake + STT)
    python main.py --test-stt   → Test just the speech-to-text
    python main.py --calibrate  → Calibrate mic for your room
    python main.py --devices    → List available microphones
"""

import sys
import config
from backend.voice.tts import speak
from backend.utils.logger import get_logger
from backend.utils.status import show_banner, show_state

logger = get_logger("MAIN")


def start_syn():
    """Boot sequence — initialize all systems and start listening."""
    show_banner()
    show_state("BOOTING")

    # Boot greeting
    speak(config.SYN_BOOT_MESSAGE)
    speak("Voice systems online. Clap detection active.")
    logger.info("Boot sequence complete.")

    # Instructions
    print(f"  [*] {config.CLAP_PATTERN.capitalize()}-clap to wake SYN")
    print("  [*] Speak your command after the greeting")
    print("  [*] Press Ctrl+C to shut down")
    print()

    # Start the full pipeline
    from backend.voice.listen_loop import ListenLoop
    loop = ListenLoop()

    try:
        loop.start()
    except KeyboardInterrupt:
        loop.stop()
        show_state("SHUTDOWN")
        speak("Shutting down. Goodbye.")
        logger.info("SYN shut down by user.")
        print()


def test_stt():
    """Quick test: record + transcribe without clap detection."""
    print()
    print("=" * 55)
    print("  S.Y.N. -- Speech-to-Text Test")
    print("  Speak after the prompt, then go silent.")
    print("=" * 55)
    print()

    from backend.voice.audio_recorder import AudioRecorder
    from backend.voice.stt import SpeechToText

    recorder = AudioRecorder()
    stt = SpeechToText()

    print("  Say something...")
    wav_file = recorder.record()

    if wav_file:
        result = stt.transcribe(wav_file)
        print()
        print("  " + "-" * 50)
        print(f"  >> You said: \"{result['text']}\"")
        print(f"     Language: {result['language']}")
        print(f"     Confidence: {result['confidence']}")
        print("  " + "-" * 50)
        AudioRecorder.cleanup(wav_file)
    else:
        print("  No speech detected.")


def calibrate():
    """Run mic calibration for your room."""
    from backend.voice.clap_detector import calibrate_clap_threshold
    calibrate_clap_threshold()


def list_devices():
    """List all available audio input devices."""
    from backend.voice.mic_stream import MicStream
    mic = MicStream()
    devices = mic.list_devices()

    print()
    print("  Available microphones:")
    print("  " + "-" * 40)
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']}")
        print(f"      Channels: {dev['channels']}, Rate: {dev['sample_rate']} Hz")
    print("  " + "-" * 40)
    print()
    print("  Set MIC_DEVICE_INDEX in config.py to use a specific mic.")
    print()


if __name__ == "__main__":
    if "--test-stt" in sys.argv:
        test_stt()
    elif "--calibrate" in sys.argv:
        calibrate()
    elif "--devices" in sys.argv:
        list_devices()
    else:
        start_syn()