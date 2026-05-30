"""
S.Y.N. — Audio and TTS Diagnostics Tool
=========================================
Checks the Text-to-Speech voices available on your Windows system
and runs a test speech sequence to verify audio routing.

RUN WITH:
    python -m backend.utils.diagnose_tts
"""

import sys
import pyttsx3

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_diagnostics():
    print()
    print(f"{BOLD}{CYAN}==================================================")
    print("         S.Y.N. -- TTS SPEECH DIAGNOSTICS         ")
    print(f"=================================================={RESET}")
    print()
    
    print("[*] Initializing pyttsx3 voice engine...")
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"{RED}[FAIL] Could not initialize pyttsx3: {e}{RESET}")
        print("This typically happens if the Windows Audio service is stopped or registry keys are corrupted.")
        sys.exit(1)
        
    print(f"{GREEN}[SUCCESS] Voice engine initialized.{RESET}")
    print()
    
    voices = engine.getProperty("voices")
    print(f"{BOLD}Available System Voices ({len(voices)}):{RESET}")
    print("-" * 65)
    for idx, voice in enumerate(voices):
        print(f"  Voice ID [{idx}]:")
        print(f"    - Name: {voice.name}")
        print(f"    - ID: {voice.id}")
        print(f"    - Languages: {voice.languages}")
        print(f"    - Gender: {voice.gender}")
        print()
        
    print("-" * 65)
    print()
    
    print(f"{BOLD}Testing Voice Audio Output:{RESET}")
    print("We will attempt to speak a test sentence with each voice.")
    print("Please listen closely to hear which voice index plays through your speakers/headphones.")
    print()
    
    for idx, voice in enumerate(voices):
        print(f"[{idx}] Attempting to speak using voice: {voice.name}...")
        engine.setProperty("voice", voice.id)
        engine.setProperty("rate", 175)
        
        test_text = f"Hello. This is voice number {idx}. Can you hear me, Boss?"
        print(f"    Speaking: \"{test_text}\"")
        try:
            engine.say(test_text)
            engine.runAndWait()
            print(f"    {GREEN}[DONE] Speech command completed.{RESET}")
        except Exception as e:
            print(f"    {RED}[FAIL] Error during speech execution: {e}{RESET}")
        print()
        
    print(f"{BOLD}How to fix audio issues:{RESET}")
    print(f"1. If you heard one of the voices: Update {YELLOW}TTS_VOICE_INDEX{RESET} in your {YELLOW}config.py{RESET} to that index.")
    print("2. If you heard absolutely nothing:")
    print("   - Check that your default Windows Playback Device is active and volume is turned up.")
    print("   - Make sure your headphones/speakers are not routed to a virtual output (like NVIDIA Broadcast virtual output or VoiceMeeter) that is muted.")
    print()


if __name__ == "__main__":
    run_diagnostics()
