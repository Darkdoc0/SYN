"""
S.Y.N. Brain — Module Registry
===============================
Provides a dynamic registration system for S.Y.N.'s capability modules.
This enables clean decoupling: new capabilities (like app launching, smart home,
or weather lookups) can register themselves without editing the central routing code.

DESIGN PATTERNS:
1. Decorator Pattern: `@register_module(intent_name)` maps functions to intents dynamically.
2. Registry Pattern: Centralizes module bindings for the dispatcher to query.
"""

from typing import Dict, Any, Callable
from backend.utils.logger import get_logger

logger = get_logger("REGISTRY")

# Dictionary mapping intent names (keys) to handler functions (values)
_REGISTRY: Dict[str, Callable[[Dict[str, Any]], str]] = {}


def register_module(intent: str):
    """
    Python Decorator to register a function as an executor for a specific intent.
    
    Example:
        @register_module("weather")
        def get_weather(entities):
            return "It is sunny in Paris."
    """
    def decorator(func: Callable[[Dict[str, Any]], str]):
        _REGISTRY[intent] = func
        logger.info(f"Registered capability module: intent '{intent}' -> function '{func.__name__}'")
        return func
    return decorator


def get_handler(intent: str) -> Callable[[Dict[str, Any]], str]:
    """Retrieves the registered handler function for a given intent."""
    return _REGISTRY.get(intent)


# ─────────────────────────────────────────────────────────────
#  MOCK CAPABILITY HANDLERS (Placeholder for Phase 4+)
# ─────────────────────────────────────────────────────────────

@register_module("open_app")
def handle_open_app(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 8: App Launcher."""
    app_name = entities.get("app_name", "unknown application")
    action = entities.get("action", "open")
    
    logger.info(f"Mocking open_app: {action} {app_name}")
    if action in ["close", "exit", "kill"]:
        return f"Mock Action: Closing {app_name}."
    return f"Mock Action: Launching {app_name}."


@register_module("search_web")
def handle_search_web(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 10: Web Search."""
    query = entities.get("query", "nothing")
    logger.info(f"Mocking search_web: query='{query}'")
    return f"Mock Action: Searching the web for {query}."


@register_module("play_music")
def handle_play_music(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 14: Music Playback."""
    song = entities.get("song_name")
    artist = entities.get("artist_name")
    action = entities.get("action")
    
    logger.info(f"Mocking play_music: song='{song}', artist='{artist}', action='{action}'")
    if action:
        return f"Mock Action: Music controls, executing {action}."
        
    response = f"Mock Action: Playing {song}"
    if artist:
        response += f" by {artist}"
    return response + "."


@register_module("system_cmd")
def handle_system_cmd(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 9: System Control."""
    action = entities.get("action")
    percent = entities.get("volume_percent")
    
    logger.info(f"Mocking system_cmd: action='{action}', percent={percent}")
    if action == "volume_up":
        return "Mock Action: Increasing volume."
    elif action == "volume_down":
        return "Mock Action: Decreasing volume."
    elif action == "mute":
        return "Mock Action: Muting system volume."
    elif action == "unmute":
        return "Mock Action: Unmuting system volume."
    elif action == "set" and percent is not None:
        return f"Mock Action: Setting volume to {percent} percent."
    elif action:
        return f"Mock Action: Executing computer {action} sequence."
        
    return "Mock Action: Received unknown system instruction."


@register_module("weather")
def handle_weather(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 15: Weather Briefing."""
    location = entities.get("location", "your location")
    time = entities.get("time", "today")
    
    logger.info(f"Mocking weather: location='{location}', time='{time}'")
    return f"Mock Action: Fetching the weather forecast for {location} for {time}."


@register_module("todo")
def handle_todo(entities: Dict[str, Any]) -> str:
    """Mock handler for Day 11: To-Do lists."""
    action = entities.get("action")
    task = entities.get("todo_task", "")
    
    logger.info(f"Mocking todo: action='{action}', task='{task}'")
    if action == "add":
        return f"Mock Action: Adding task, {task}, to your todo list."
    elif action == "list":
        return "Mock Action: Reading back your active todo list."
    elif task:
        return f"Mock Action: Modifying task, {task}."
        
    return "Mock Action: Accessing todo registry."
