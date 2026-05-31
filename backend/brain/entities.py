"""
S.Y.N. Brain — Entity Extraction & Text Cleaning
=================================================
Provides text normalization, sanitization, and helper functions to extract
structured entities from raw transcribed user speech.

WHAT WE DO:
1. Normalization: Lowercase everything, strip punctuation (?, !, ., ,).
2. Filler Removal: Strip leading/trailing verbal fluff like "hey syn", "please", "can you".
3. Group Extraction: Parse named capture groups from regular expression matches.
"""

import re
from backend.utils.logger import get_logger

logger = get_logger("ENTITIES")

# Common verbal filler patterns to strip from the beginning or end of spoken queries
# so they don't break our strict pattern matching.
LEADING_FILLERS = re.compile(
    r"^(?:hey\s+syn|syn|please|can\s+you|could\s+you(?:\s+please)?|would\s+you\s+mind|go\s+ahead\s+and)\s+",
    re.IGNORECASE
)

TRAILING_FILLERS = re.compile(
    r"\s+(?:please|thank\s+you|thanks|now|for\s+me)$",
    re.IGNORECASE
)

PUNCTUATION_TO_STRIP = re.compile(r"[?!\.,]")


def clean_query(raw_query: str) -> str:
    """
    Clean and normalize raw voice transcription.
    
    Example:
        "Hey SYN, could you please open Google Chrome?" -> "open google chrome"
    """
    if not raw_query:
        return ""
        
    # 1. Convert to lowercase & strip trailing/leading whitespace
    text = raw_query.strip().lower()
    
    # 2. Remove common voice transcription punctuation
    text = PUNCTUATION_TO_STRIP.sub("", text)
    
    # 3. Strip leading verbal fillers
    text = LEADING_FILLERS.sub("", text)
    
    # 4. Strip trailing verbal fillers
    text = TRAILING_FILLERS.sub("", text)
    
    # 5. Collapse multiple spaces to single spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    logger.debug(f"Normalized query: '{raw_query}' -> '{text}'")
    return text


def extract_entities(match: re.Match) -> dict:
    """
    Extracts all named capture groups from a successful regex Match object.
    Strips trailing and leading whitespaces from any captured values.
    
    Example:
        Pattern: "open (?P<app_name>.+)"
        Match: "open google chrome"
        Returns: {"app_name": "google chrome"}
    """
    entities = {}
    if match:
        # groupdict() returns a dictionary of all named capture groups in the match
        raw_groups = match.groupdict()
        for key, val in raw_groups.items():
            if val is not None:
                # Clean up extracted entity value (strip spaces, etc.)
                entities[key] = val.strip()
                
    return entities


def post_process_entities(intent: str, entities: dict) -> dict:
    """
    Standardizes and normalizes entity values based on the intent category.
    This helps the downstream action executors receive uniform variables.
    
    Example:
        For 'system_cmd', if action is 'volume' and value is 'up',
        we normalize it to standard keys.
    """
    processed = entities.copy()
    
    # 1. Normalize app commands
    if intent == "open_app":
        action = processed.get("app_action")
        if action in ["close", "exit", "kill", "terminate"]:
            processed["action"] = "close"
        else:
            processed["action"] = "open"
            
    # 2. Normalize system commands
    elif intent == "system_cmd":
        action = processed.get("system_action")
        val = processed.get("volume_value")
        
        # If user said "volume up", map to standard action
        if action in ["up", "down"]:
            processed["action"] = f"volume_{action}"
        elif action:
            processed["action"] = action
            
        if val:
            try:
                processed["volume_percent"] = int(val)
                if not processed.get("action"):
                    processed["action"] = "set"
            except ValueError:
                pass
                
    # 2. Normalize music commands
    elif intent == "play_music":
        control = processed.get("music_control")
        if control:
            processed["action"] = control
            
    # 3. Normalize todo commands
    elif intent == "todo":
        # Check if the user is asking to list vs add
        if "todo_task" not in processed:
            # If no task is present, the action is likely "list"
            processed["action"] = "list"
        else:
            processed["action"] = "add" # default action if task present
            
    return processed
