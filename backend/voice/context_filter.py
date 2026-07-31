"""
S.Y.N. — Context Filter (JARVIS Engine)
=======================================
Decides whether S.Y.N. should respond to a transcribed piece of text.
Replaces the physical clap detector with intelligent context awareness.
"""

import time
import re
import config
from backend.utils.logger import get_logger
from backend.brain.intent_router import IntentResult

logger = get_logger("CONTEXT")

class ContextFilter:
    def __init__(self):
        self.wake_words = getattr(config, "WAKE_WORDS", ["syn", "sin", "jarvis"])
        self.conversation_window = getattr(config, "CONVERSATION_WINDOW", 15.0)
        self.bypass_threshold = getattr(config, "COMMAND_BYPASS_THRESHOLD", 0.85)
        
        self.last_assistant_speech_time = 0.0

    def mark_assistant_spoke(self):
        """Call this whenever S.Y.N. finishes speaking to open the conversation window."""
        self.last_assistant_speech_time = time.time()
        logger.debug(f"Conversation window opened for {self.conversation_window}s")

    def _strip_wake_word(self, text: str) -> tuple[bool, str]:
        """
        Checks if the text contains a wake word.
        If it does, returns (True, text_without_wake_word).
        Otherwise returns (False, original_text).
        """
        text_lower = text.lower()
        
        # Check if any wake word is in the text
        for wake_word in self.wake_words:
            wake_word = wake_word.lower()
            # Regex to match the wake word as a distinct word (accounting for punctuation)
            pattern = r'\b' + re.escape(wake_word) + r'\b'
            
            if re.search(pattern, text_lower):
                # We found a wake word! Remove the first instance of it from the string.
                # We do this carefully to preserve original casing in the rest of the string.
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    start, end = match.span()
                    clean_text = text[:start] + text[end:]
                    # Clean up any leftover double spaces or leading/trailing spaces
                    clean_text = " ".join(clean_text.split()).strip()
                    # If they just said "SYN", return empty string
                    return True, clean_text
                    
        return False, text

    def should_respond(self, text: str, intent_router, overlapped_with_assistant: bool = False) -> tuple[bool, str, IntentResult | None]:
        """
        Analyzes the text to decide if S.Y.N. should process it.
        
        Rules:
        1. Explicit Wake Word -> Always Respond
        2. Conversation Window -> Always Respond
        3. Direct Command Bypass -> Respond if Intent Confidence >= threshold
        
        Returns:
            (should_respond, clean_text, intent_result_if_already_routed)
        """
        has_wake_word, clean_text = self._strip_wake_word(text)
        
        # Rule 1: Explicit Wake Word
        if has_wake_word:
            logger.info("Context: Wake word detected.")
            # We don't route yet, the ListenLoop will do it with the clean_text
            return True, clean_text, None

        if overlapped_with_assistant:
            # If the microphone picked this up while S.Y.N. was speaking, it is almost
            # certainly her own voice echoing through the mic. Unless she explicitly said
            # her own wake word (handled above), we completely ignore it.
            logger.info(f"Context: Ignored echo/overlap -> '{text}'")
            return False, text, None

        # Rule 2: Conversation Window (S.Y.N. just spoke recently)
        time_since_speech = time.time() - self.last_assistant_speech_time
        if time_since_speech <= self.conversation_window:
            logger.info(f"Context: Inside conversation window ({time_since_speech:.1f}s ago). Responding.")
            return True, clean_text, None

        # Rule 3: Direct Command Bypass (High confidence intent without wake word)
        # We temporarily route the raw text to see if it's a valid command
        route_result = intent_router.route(clean_text)
        
        if route_result.intent != "chat" and route_result.confidence >= self.bypass_threshold:
            logger.info(f"Context: Direct command bypass triggered (Intent: {route_result.intent}, Conf: {route_result.confidence}).")
            return True, clean_text, route_result

        # Rule 4: Ignore Background Noise
        logger.debug(f"Context: Ignored background speech -> '{text}'")
        return False, text, None
