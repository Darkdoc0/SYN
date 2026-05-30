"""
S.Y.N. Brain — Personality Loader
==================================
Manages S.Y.N.'s identity, system prompt loading, and dynamic context injection
(like the user's name, current time, and operating system).

DESIGN PATTERNS:
1. Singleton/Factory: Standardized loading of prompt assets.
2. Dynamic Context Injection: Keeps the LLM aware of temporal and environmental state.
"""

import os
import sys
import platform
import getpass
from datetime import datetime
from backend.utils.logger import get_logger
import config

logger = get_logger("PERSONALITY")


def get_time_of_day() -> str:
    """Returns 'morning', 'afternoon', or 'evening' based on current hour."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    else:
        return "evening"


def load_system_prompt() -> str:
    """
    Loads S.Y.N.'s base system prompt and appends dynamic environment parameters.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "prompts", "system.txt")
    
    # Default fallback prompt if the file cannot be loaded
    base_prompt = (
        "You are S.Y.N., a witty, sarcastic, JARVIS-like personal assistant. "
        "Keep responses brief, conversational, and optimize for text-to-speech (no markdown)."
    )
    
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                base_prompt = f.read().strip()
            logger.debug("Successfully loaded system prompt template from system.txt")
        except Exception as e:
            logger.error(f"Error reading system prompt file: {e}. Using fallback.")
            
    # Collect dynamic runtime details to ground the LLM
    current_time = datetime.now().strftime("%I:%M %p")
    current_date = datetime.now().strftime("%B %d, %Y")
    user_name = getpass.getuser()
    os_name = platform.system()
    os_release = platform.release()
    time_greeting = get_time_of_day()
    
    # Inject active session variables at the end of the system instructions
    dynamic_context = (
        f"\n\nACTIVE SESSION PARAMETERS:\n"
        f"- Assistant Name: {config.SYN_NAME}\n"
        f"- Current Time: {current_time} ({time_greeting})\n"
        f"- Current Date: {current_date}\n"
        f"- User Active: {user_name}\n"
        f"- Operating System: {os_name} (Build {os_release})\n"
    )
    
    full_prompt = base_prompt + dynamic_context
    return full_prompt
