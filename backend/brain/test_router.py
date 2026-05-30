"""
S.Y.N. Brain — Intent Router Test Suite
=======================================
Runs a comprehensive battery of tests against the IntentRouter to ensure
pattern matching, entity extraction, and fallbacks are working flawlessly.

RUN WITH:
    python -m backend.brain.test_router
"""

import sys
from backend.brain.intent_router import IntentRouter


# ANSI Color Codes for terminal beauty (Day 4 standard)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_tests():
    """Run all assertion checks against the IntentRouter."""
    print()
    print(f"{BOLD}{CYAN}==================================================")
    print("      S.Y.N. -- INTENT ROUTER VERIFICATION TEST   ")
    print(f"=================================================={RESET}")
    print()
    
    try:
        router = IntentRouter()
    except Exception as e:
        print(f"{RED}[FAIL] Could not initialize IntentRouter: {e}{RESET}")
        sys.exit(1)

    # Test database matrix: (Raw Query, Expected Intent, Target Entities)
    test_cases = [
        # 1. App launching
        ("open google chrome", "open_app", {"app_name": "google chrome"}),
        ("please start notepad", "open_app", {"app_name": "notepad"}),
        ("terminate vscode please", "open_app", {"app_name": "vscode"}),
        ("kill firefox now", "open_app", {"app_name": "firefox"}),
        
        # 2. Web search
        ("search for python list comprehension tutorials", "search_web", {"query": "python list comprehension tutorials"}),
        ("google quantum computing progress", "search_web", {"query": "quantum computing progress"}),
        ("what is recipe for dark roast coffee", "search_web", {"query": "recipe for dark roast coffee"}),
        
        # 3. Music
        ("play Starboy by The Weeknd", "play_music", {"song_name": "starboy", "artist_name": "the weeknd"}),
        ("play synthwave study beats", "play_music", {"song_name": "synthwave study beats"}),
        ("pause music", "play_music", {"action": "pause"}),
        ("skip song", "play_music", {"action": "skip"}),
        
        # 4. System controls
        ("mute the sound please", "system_cmd", {"action": "mute"}),
        ("volume up", "system_cmd", {"action": "volume_up"}),
        ("set audio volume to 50%", "system_cmd", {"action": "set", "volume_percent": 50}),
        ("lock my PC", "system_cmd", {"action": "lock"}),
        ("shutdown computer now", "system_cmd", {"action": "shutdown"}),
        
        # 5. Weather
        ("weather in London", "weather", {"location": "london"}),
        ("what is the weather forecast for Mumbai?", "weather", {"location": "mumbai"}),
        ("will it rain tomorrow in Paris", "weather", {"location": "paris", "time": "tomorrow"}),
        ("will it rain today", "weather", {"time": "today"}),
        
        # 6. Todo list
        ("add buy whole milk to my todo list", "todo", {"todo_task": "buy whole milk", "action": "add"}),
        ("remind me to stretch at 5 PM to my tasks", "todo", {"todo_task": "stretch at 5 pm", "action": "add"}),
        ("show my todo list", "todo", {"action": "list"}),
        ("delete task buy whole milk", "todo", {"todo_task": "buy whole milk"}),
        
        # 7. Keyword fallbacks (heuristics testing - confidence should be 0.6)
        ("show weather details", "weather", {}),
        ("tell me todo statistics", "todo", {}),
        ("google something", "search_web", {}),
        
        # 8. Chat Fallback (no patterns/keywords match - confidence should be 0.0)
        ("hello, how is your day going?", "chat", {}),
        ("what is the meaning of life, universe and everything?", "chat", {})
    ]

    total = len(test_cases)
    passed = 0
    
    # Header format
    print(f"{BOLD}{'RAW INPUT':<45} | {'INTENT':<12} | {'CONF':<4} | {'STATUS':<5}{RESET}")
    print("-" * 75)

    for query, expected_intent, expected_entities in test_cases:
        res = router.route(query)
        
        # Validate intent match
        intent_ok = res.intent == expected_intent
        
        # Validate confidence score bounds
        conf_ok = True
        if res.confidence == 1.0:
            # Full regex match
            pass
        elif res.confidence == 0.6:
            # Heuristic match
            pass
        elif res.confidence == 0.0:
            # Fallback
            pass
        else:
            conf_ok = False
            
        # Validate entities match subset
        entities_ok = True
        for k, v in expected_entities.items():
            if res.entities.get(k) != v:
                entities_ok = False
                break
                
        case_passed = intent_ok and conf_ok and entities_ok
        
        # Display feedback formatting
        if case_passed:
            passed += 1
            status_str = f"{GREEN}PASS{RESET}"
        else:
            status_str = f"{RED}FAIL{RESET}"
            
        # Print status summary row
        truncated_query = query if len(query) <= 45 else query[:42] + "..."
        print(f"{truncated_query:<45} | {res.intent:<12} | {res.confidence:<4.1f} | {status_str}")
        
        # If test fails, show diagnostic detail
        if not case_passed:
            print(f"   {YELLOW}>>> Expected: intent='{expected_intent}', entities={expected_entities}{RESET}")
            print(f"   {YELLOW}>>> Got:      intent='{res.intent}', confidence={res.confidence}, entities={res.entities}{RESET}")
            print()

    print("-" * 75)
    print()
    
    score_color = GREEN if passed == total else YELLOW if passed > 0 else RED
    print(f"{BOLD}Test Result Summary: {score_color}{passed}/{total} Passed{RESET}")
    print()
    
    if passed == total:
        print(f"{BOLD}{GREEN}[SUCCESS] All intent routing assertions successfully verified!{RESET}")
        sys.exit(0)
    else:
        print(f"{BOLD}{RED}[ERROR] Some routing test assertions failed. Please debug patterns.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
