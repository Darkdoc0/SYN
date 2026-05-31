"""
S.Y.N. Automation — App Launcher
================================
Handles opening and closing applications.
Uses an app_registry.json for hardcoded matches, and auto-discovers
installed applications by scanning the Windows Start Menu.
"""

import os
import glob
import json
import psutil
import difflib
from typing import Dict, Any, Optional, Tuple

from backend.utils.logger import get_logger
from backend.brain.module_registry import register_module

logger = get_logger("APP_LAUNCHER")

def _get_registry_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "app_registry.json")

def load_app_registry() -> Dict[str, str]:
    path = _get_registry_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load app registry: {e}")
    return {}

def discover_start_menu_apps() -> Dict[str, str]:
    """Scans the Windows Start Menu for application shortcuts (.lnk)."""
    apps = {}
    
    # Common start menu locations on Windows
    locations = [
        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs")
    ]
    
    for loc in locations:
        if not os.path.exists(loc):
            continue
            
        # Recursively search for .lnk files
        for root, _, files in os.walk(loc):
            for file in files:
                if file.lower().endswith(".lnk"):
                    name = file[:-4].lower() # strip .lnk
                    # Clean up common fluff in shortcut names
                    name = name.replace(" (x86)", "").replace(" (64-bit)", "")
                    full_path = os.path.join(root, file)
                    apps[name] = full_path
                    
    return apps

def find_best_app_match(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds the best matching application path for a given query.
    Returns a tuple of (matched_name, executable_path).
    """
    query = query.lower().strip()
    
    # 1. Check custom registry first (exact or close match)
    registry = load_app_registry()
    matches = difflib.get_close_matches(query, registry.keys(), n=1, cutoff=0.7)
    if matches:
        return matches[0], registry[matches[0]]
        
    # 2. Check auto-discovered start menu apps
    start_menu_apps = discover_start_menu_apps()
    matches = difflib.get_close_matches(query, start_menu_apps.keys(), n=1, cutoff=0.6)
    if matches:
        return matches[0], start_menu_apps[matches[0]]
        
    # 3. Fallback to just returning the query as an executable (e.g. 'notepad' -> 'notepad.exe' via PATH)
    return query, query

def close_application(query: str) -> str:
    """Attempts to kill processes matching the app query."""
    query_lower = query.lower().replace(" ", "")
    killed_count = 0
    
    # Check custom registry to get exact process name if possible
    registry = load_app_registry()
    matches = difflib.get_close_matches(query_lower, registry.keys(), n=1, cutoff=0.7)
    target_exe = None
    if matches:
        target_exe = registry[matches[0]].lower()
        if not target_exe.endswith(".exe"):
            target_exe = None
    
    # Iterate through running processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            p_name = proc.info['name']
            if not p_name:
                continue
                
            p_name_lower = p_name.lower()
            
            # Match by exact exe name from registry, or substring match against query
            is_match = False
            if target_exe and target_exe == p_name_lower:
                is_match = True
            elif query_lower in p_name_lower.replace(".exe", ""):
                is_match = True
                
            if is_match:
                logger.info(f"Terminating process {p_name} (PID: {proc.info['pid']})")
                proc.terminate()
                killed_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    if killed_count > 0:
        return f"I have closed {query}."
    else:
        return f"I couldn't find any running instances of {query} to close."


@register_module("open_app")
def handle_open_app(entities: Dict[str, Any]) -> str:
    """
    Handler for opening and closing applications.
    Triggered by queries like 'open Chrome' or 'close Notepad'.
    """
    app_name = entities.get("app_name")
    action = entities.get("action", "open")
    
    if not app_name:
        return "I'm not sure which application you want me to interact with, Boss."
        
    logger.info(f"App Launcher executing: {action} on '{app_name}'")
    
    if action == "close":
        return close_application(app_name)
        
    # Open action
    matched_name, exe_path = find_best_app_match(app_name)
    
    try:
        if exe_path.endswith(".exe") or exe_path.endswith(".lnk") or ":" in exe_path:
             os.startfile(exe_path)
        else:
             # Just try to launch the raw query in shell (e.g. 'calc')
             os.startfile(app_name)
             
        # Format the name nicely for speech
        display_name = matched_name.title() if matched_name else app_name.title()
        return f"Opening {display_name} for you, Boss."
        
    except Exception as e:
        logger.error(f"Failed to launch '{app_name}': {e}")
        return f"I encountered an error trying to launch {app_name}."
