"""
S.Y.N. Brain — LLM Engine Test Suite
====================================
Verifies personality construction, context memory retention across multiple
turns, and checks API connection/fallback structures gracefully.

RUN WITH:
    python -m backend.brain.test_llm
"""

import os
import sys
from backend.brain.personality import load_system_prompt
from backend.brain.llm_engine import LLMEngine, ConversationContext

# ANSI Color Codes for terminal beauty
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def test_personality():
    """Verify that personality loads and dynamically injects details."""
    print(f"{BOLD}[*] Testing personality prompt loader...{RESET}")
    prompt = load_system_prompt()
    
    assert "S.Y.N." in prompt or "Synthetic Yielding Nexus" in prompt, "Missing assistant name."
    assert "ACTIVE SESSION PARAMETERS:" in prompt, "Missing runtime environment variables block."
    assert "Operating System:" in prompt, "Missing OS context injection."
    
    print(f"{GREEN}[PASS] Personality system prompt verified. Dynamic context injected successfully.{RESET}")
    print()


def test_context_memory():
    """Verify context length controls and rolling history queues."""
    print(f"{BOLD}[*] Testing conversation context queue...{RESET}")
    context = ConversationContext(max_turns=2)
    
    # Add 3 turns (6 messages total). Since limit is 2 turns (4 messages), the first turn should be dropped.
    context.add_message("user", "Turn 1: hello")
    context.add_message("assistant", "Hi there")
    context.add_message("user", "Turn 2: my favorite color is green")
    context.add_message("assistant", "Nice choice")
    context.add_message("user", "Turn 3: how is the weather?")
    context.add_message("assistant", "It's sunny")
    
    messages = context.get_messages(system_prompt="system instructions")
    
    # messages[0] = system prompt
    # messages[1:] = rolling history (4 messages max)
    assert len(messages) == 5, f"Expected 5 messages (1 system + 4 history), got {len(messages)}"
    assert messages[1]["content"] == "Turn 2: my favorite color is green", "Context failed to drop oldest turn."
    assert messages[4]["content"] == "It's sunny", "Context failed to retain latest turn."
    
    print(f"{GREEN}[PASS] Conversation context queue verified. Rolling memory operates correctly.{RESET}")
    print()


def test_llm_execution():
    """Tests actual LLM call and fallback cascade routing."""
    print(f"{BOLD}[*] Testing LLM execution and fallbacks...{RESET}")
    
    engine = LLMEngine()
    
    # We will attempt a generation. Since the user might not have Ollama running
    # or cloud API keys configured yet, we will catch errors gracefully
    # and explain exactly what is active or missing.
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    print(f"    Current API Key States:")
    print(f"      - OPENAI_API_KEY: {'{GREEN}SET{RESET}' if openai_key else '{YELLOW}NOT SET{RESET}'}")
    print(f"      - ANTHROPIC_API_KEY: {'{GREEN}SET{RESET}' if anthropic_key else '{YELLOW}NOT SET{RESET}'}")
    print()
    
    print(f"    Attempting generation query: 'Hello, who are you?'")
    try:
        reply = engine.generate("Hello, who are you?")
        print()
        print(f"    {BOLD}S.Y.N. Response:{RESET}")
        print(f"    {CYAN}\"{reply}\"{RESET}")
        print()
        
        # Verify response isn't completely empty
        assert len(reply) > 5, "Response is too short or empty."
        print(f"{GREEN}[PASS] LLM generated response successfully!{RESET}")
        
    except Exception as e:
        print(f"{RED}[FAIL] LLM engine encountered an unhandled execution error: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    print()
    print(f"{BOLD}{CYAN}==================================================")
    print("        S.Y.N. -- LLM ENGINE VERIFICATION TEST    ")
    print(f"=================================================={RESET}")
    print()
    
    test_personality()
    test_context_memory()
    test_llm_execution()
    
    print()
    print(f"{BOLD}{GREEN}[SUCCESS] All LLM engine code checks completed!{RESET}")
    print()
