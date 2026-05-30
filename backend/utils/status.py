"""
S.Y.N. — Status Display
=========================
Pretty console status indicators for SYN's current state.

Instead of messy print() calls, this gives SYN a clean,
professional look in the terminal.

STATES:
    BOOTING     → Starting up
    READY       → Waiting for clap
    LISTENING   → Recording speech
    PROCESSING  → Transcribing / thinking
    RESPONDING  → Speaking response
    ERROR       → Something went wrong
    SHUTDOWN    → Shutting down
"""

from backend.utils.logger import get_logger

logger = get_logger("STATUS")


# ── State display strings ──
# Using ASCII-safe characters for Windows compatibility
STATES = {
    "BOOTING":    "[>>>] BOOTING",
    "READY":      "[ * ] READY — waiting for clap",
    "LISTENING":  "[ @ ] LISTENING — speak now...",
    "PROCESSING": "[...] PROCESSING — transcribing...",
    "THINKING":   "[...] THINKING — generating response...",
    "RESPONDING":  "[ > ] RESPONDING",
    "ERROR":      "[ERR] ERROR",
    "SHUTDOWN":   "[OFF] SHUTTING DOWN",
}


def show_state(state: str, detail: str = ""):
    """
    Display SYN's current state in the console.

    Args:
        state: One of the STATES keys
        detail: Optional extra info to show
    """
    display = STATES.get(state.upper(), f"[???] {state}")

    if detail:
        display += f" — {detail}"

    # Use the separator line for visibility
    print()
    print(f"  {display}")
    print()

    logger.debug(f"State changed: {state} {detail}")


def show_transcription(text: str, language: str = "", confidence: float = 0.0):
    """Display transcription result in a formatted box."""
    print()
    print("  " + "-" * 50)
    print(f"  >> You said: \"{text}\"")
    if language:
        print(f"     Language: {language} | Confidence: {confidence}")
    print("  " + "-" * 50)
    print()


def show_banner():
    """Display the SYN boot banner."""
    print()
    print("=" * 55)
    print("   ███████╗██╗   ██╗███╗   ██╗")
    print("   ██╔════╝╚██╗ ██╔╝████╗  ██║")
    print("   ███████╗ ╚████╔╝ ██╔██╗ ██║")
    print("   ╚════██║  ╚██╔╝  ██║╚██╗██║")
    print("   ███████║   ██║   ██║ ╚████║")
    print("   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝")
    print("   Synthetic Yielding Nexus v0.3")
    print("=" * 55)
    print()
