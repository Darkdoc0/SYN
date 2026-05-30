"""
S.Y.N. Brain — Integrated Pipeline Simulation
=============================================
Simulates the core voice loop (text input -> routing -> dispatch -> TTS response)
without requiring clap triggers or microphone capture.

RUN WITH:
    python -m backend.brain.test_integration
"""

import sys
from backend.brain.intent_router import IntentRouter
from backend.brain.dispatcher import Dispatcher
from backend.voice.tts import speak

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def simulate_pipeline():
    print()
    print(f"{BOLD}{CYAN}==================================================")
    print("      S.Y.N. -- PIPELINE INTEGRATION TEST         ")
    print(f"=================================================={RESET}")
    print()
    
    print("[*] Initializing Intent Router...")
    router = IntentRouter()
    
    print("[*] Initializing Command Dispatcher...")
    dispatcher = Dispatcher()
    
    # Test cases to simulate
    queries = [
        "please open Google Chrome",
        "set the volume to 80 percent",
        "will it rain tomorrow in Paris",
        "who are you?" # This should cascade to the conversational LLM / offline fallback
    ]
    
    print()
    print(f"{BOLD}Running Simulation Cascade:{RESET}")
    print("-" * 60)
    
    for idx, query in enumerate(queries, 1):
        print(f"\n{BOLD}Test {idx}: User Spoke: \"{query}\"{RESET}")
        
        # 1. Route Intent
        result = router.route(query)
        print(f"   -> Routed Intent: '{result.intent}' (Conf: {result.confidence})")
        print(f"   -> Extracted Variables: {result.entities}")
        
        # 2. Dispatch to module executor
        response_text = dispatcher.dispatch(result)
        print(f"   -> Dispatch Response: \"{response_text}\"")
        
        # 3. Speak response (Audio Check)
        print(f"   -> {YELLOW}[Audio Output] Speaking...{RESET}")
        speak(response_text)
        
    print()
    print("-" * 60)
    print(f"{BOLD}{GREEN}[SUCCESS] Pipeline integration test executed successfully!{RESET}")
    print()


if __name__ == "__main__":
    try:
        simulate_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"{RED}[FAIL] Integration pipeline failed with error: {e}{RESET}")
        sys.exit(1)
