"""
S.Y.N. Brain — Intent Router (NLP Engine)
=========================================
Determines the user's intent from clean transcribed text and extracts
the required variables (entities) to route the command to the right system.

DESIGN PATTERNS USED:
1. Pattern Matching: Regex-based strict matching using pre-compiled expressions.
2. Heuristics: Fallback keyword overlap calculations to identify intents.
3. Decoupling: Separates query classification logic from actions execution.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from backend.utils.logger import get_logger
from backend.brain.entities import clean_query, extract_entities, post_process_entities

logger = get_logger("ROUTER")


@dataclass
class IntentResult:
    """Standardized representation of routing classification."""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    raw_query: str
    clean_query: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


# Default fallback keywords for keyword-based heuristic routing
FALLBACK_KEYWORDS = {
    "open_app": ["open", "start", "launch", "run", "close", "kill", "exit"],
    "search_web": ["search", "google", "lookup", "look up", "find"],
    "play_music": ["play", "music", "song", "audio", "spotify", "pause", "resume", "skip", "next"],
    "system_cmd": ["volume", "sound", "mute", "unmute", "brightness", "lock", "shutdown", "restart", "sleep"],
    "weather": ["weather", "temperature", "forecast", "rain", "sunny", "hot", "cold", "degree"],
    "todo": ["todo", "to-do", "task", "tasks", "remind", "reminder", "list"]
}


class IntentRouter:
    """
    NLP routing engine that matches clean user input to system intents
    using pre-compiled regular expression patterns and keyword fallback rules.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Resolve path to intents.json dynamically relative to this script
        if not config_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "intents.json")
            
        self.config_path = config_path
        self.intents: Dict[str, List[re.Pattern]] = {}
        self.load_intents()

    def load_intents(self):
        """Load and compile regular expressions from intents.json."""
        logger.info(f"Loading patterns database from {self.config_path}")
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.intents = {}
            for intent_name, patterns in data.items():
                compiled_patterns = []
                for p in patterns:
                    try:
                        # Compile to regex object (case-insensitive for robust matching)
                        compiled_patterns.append(re.compile(p, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Failed to compile pattern '{p}' for intent '{intent_name}': {e}")
                self.intents[intent_name] = compiled_patterns
                
            logger.info(f"Successfully compiled {sum(len(p) for p in self.intents.values())} patterns across {len(self.intents)} intents.")
        except Exception as e:
            logger.critical(f"Critical failure loading intents database: {e}")
            raise e

    def _match_patterns(self, query: str) -> Optional[tuple]:
        """
        Check query against compiled regex patterns.
        Returns: (intent_name, match_object) if found, else None
        """
        for intent_name, patterns in self.intents.items():
            for pattern in patterns:
                # search allows matching patterns within the string (more flexible than match)
                match = pattern.search(query)
                if match:
                    logger.debug(f"Pattern matched! Intent: '{intent_name}', Pattern: '{pattern.pattern}'")
                    return intent_name, match
        return None

    def _heuristic_fallback(self, query: str) -> Optional[str]:
        """
        Calculates simple keyword intersection fallback if strict patterns fail.
        Returns the intent name if key terms match, else None.
        """
        words = set(query.split())
        best_intent = None
        max_score = 0
        
        for intent_name, keywords in FALLBACK_KEYWORDS.items():
            # Check overlap between words in query and intent keywords
            overlap = words.intersection(keywords)
            score = len(overlap)
            
            # Additional boost if the query contains exact phrase matches
            for phrase in keywords:
                if " " in phrase and phrase in query:
                    score += 2
                    
            if score > max_score:
                max_score = score
                best_intent = intent_name
                
        # Return fallback intent only if we have a significant keyword signal (score >= 1)
        if best_intent and max_score >= 1:
            logger.debug(f"Keyword fallback active. Routed to '{best_intent}' (Score: {max_score})")
            return best_intent
            
        return None

    def route(self, raw_query: str) -> IntentResult:
        """
        Routes a raw string query to its corresponding intent and extracts entities.
        
        Algorithm:
        1. Clean and normalize raw query.
        2. Run regex pattern matching (High confidence: 1.0)
        3. If no pattern matches, run keyword heuristic matching (Medium confidence: 0.6)
        4. If nothing matches, fallback to chat (Low confidence: 0.0)
        """
        if not raw_query or not raw_query.strip():
            return IntentResult(
                intent="chat",
                confidence=0.0,
                entities={},
                raw_query="",
                clean_query=""
            )
            
        clean_text = clean_query(raw_query)
        
        # Phase 1: Regex Pattern Matching (High confidence)
        match_info = self._match_patterns(clean_text)
        if match_info:
            intent, match = match_info
            raw_entities = extract_entities(match)
            entities = post_process_entities(intent, raw_entities)
            return IntentResult(
                intent=intent,
                confidence=1.0,
                entities=entities,
                raw_query=raw_query,
                clean_query=clean_text
            )
            
        # Phase 2: Keyword Heuristic Matching (Medium confidence)
        fallback_intent = self._heuristic_fallback(clean_text)
        if fallback_intent:
            # We don't have safe regex capture group variables, but we can return the raw clean text
            # in a default catch-all entity container depending on the intent type.
            default_entities = {}
            if fallback_intent == "open_app":
                # Guessing that any word not in triggers could be the app name
                clean_words = [w for w in clean_text.split() if w not in FALLBACK_KEYWORDS["open_app"]]
                if clean_words:
                    default_entities["app_name"] = " ".join(clean_words)
            elif fallback_intent == "search_web":
                clean_words = [w for w in clean_text.split() if w not in FALLBACK_KEYWORDS["search_web"]]
                if clean_words:
                    default_entities["query"] = " ".join(clean_words)
            elif fallback_intent == "weather":
                clean_words = [w for w in clean_text.split() if w not in FALLBACK_KEYWORDS["weather"] and w not in ["in", "for", "at"]]
                if clean_words:
                    default_entities["location"] = " ".join(clean_words)
            elif fallback_intent == "todo":
                clean_words = [w for w in clean_text.split() if w not in FALLBACK_KEYWORDS["todo"] and w not in ["add", "my", "to"]]
                if clean_words:
                    default_entities["todo_task"] = " ".join(clean_words)
                    
            entities = post_process_entities(fallback_intent, default_entities)
            return IntentResult(
                intent=fallback_intent,
                confidence=0.6,
                entities=entities,
                raw_query=raw_query,
                clean_query=clean_text
            )
            
        # Phase 3: Conversational Fallback (LLM Chat)
        logger.debug(f"Query does not match standard patterns or keywords. Defaulting to general chat.")
        return IntentResult(
            intent="chat",
            confidence=0.0,
            entities={},
            raw_query=raw_query,
            clean_query=clean_text
        )
