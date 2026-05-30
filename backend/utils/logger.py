"""
S.Y.N. — Centralized Logging System
=====================================
All SYN modules use this logger instead of print().

PYTHON LOGGING 101 (learn this!):
─────────────────────────────────
Python's logging module has 5 severity levels:

    DEBUG    → Detailed technical info (only during development)
    INFO     → Normal operations ("Clap detected", "Recording started")
    WARNING  → Something unexpected but not fatal ("Mic latency high")
    ERROR    → Something failed ("Transcription failed")
    CRITICAL → System can't continue ("No microphone found")

Each level has a number:
    DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50

When you set level=INFO, you see INFO + WARNING + ERROR + CRITICAL
but NOT DEBUG. This lets you control verbosity.

HANDLERS — WHERE LOGS GO:
    ConsoleHandler  → prints to terminal (colored, readable)
    FileHandler     → writes to logs/syn.log (full detail, persistent)

You can have BOTH at the same time — see pretty output in terminal
AND save everything to a file for later debugging.

FORMATTERS — HOW LOGS LOOK:
    Console: "  [12:34:56] INFO  | CLAP | Clap detected! energy=950"
    File:    "2026-05-24 12:34:56,789 | INFO | backend.voice.clap_detector | Clap detected! energy=950"

Usage in any module:
    from backend.utils.logger import get_logger
    logger = get_logger("MODULE_NAME")
    logger.info("Something happened")
    logger.error("Something failed", exc_info=True)  # includes stack trace
"""

import logging
import os
import sys
from datetime import datetime
import config


# ──────────────────────────────────────────────
#  Custom Formatter — Pretty Console Output
# ──────────────────────────────────────────────
class SynConsoleFormatter(logging.Formatter):
    """
    Custom formatter that makes console logs readable and aligned.

    HOW CUSTOM FORMATTERS WORK:
        Python calls format(record) for every log message.
        `record` contains: levelname, name, message, timestamp, etc.
        We override format() to return our own styled string.
    """

    # Color codes for Windows terminal (ANSI escape sequences)
    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    def format(self, record):
        # Extract the module name (last part, e.g., "CLAP" from "SYN.CLAP")
        module = record.name.split(".")[-1]

        # Time — just HH:MM:SS for console (keep it short)
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Level — padded to 5 chars for alignment
        level = record.levelname.ljust(5)

        # Color
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        dim = self.DIM

        # Format: "  [12:34:56] INFO  | CLAP | Message here"
        formatted = (
            f"  {dim}[{time_str}]{reset} "
            f"{color}{level}{reset} "
            f"{dim}|{reset} {color}{module:>6}{reset} "
            f"{dim}|{reset} {record.getMessage()}"
        )

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


# ──────────────────────────────────────────────
#  Custom Formatter — Detailed File Output
# ──────────────────────────────────────────────
class SynFileFormatter(logging.Formatter):
    """
    Detailed formatter for the log file.
    Includes full timestamps and module paths for debugging.
    """

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ──────────────────────────────────────────────
#  Logger Factory
# ──────────────────────────────────────────────
_initialized = False


def _setup_root_logger():
    """
    Set up the root SYN logger with console + file handlers.
    Called once on first get_logger() call.

    HANDLER ARCHITECTURE:
        SYN (root logger)
        ├── ConsoleHandler → pretty colored output to terminal
        └── FileHandler    → detailed logs to logs/syn.log
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Create the root "SYN" logger
    root = logging.getLogger("SYN")
    root.setLevel(logging.DEBUG if config.DEBUG_MODE else logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if root.handlers:
        return

    # ── Console Handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if config.DEBUG_MODE else logging.INFO)
    console.setFormatter(SynConsoleFormatter())
    root.addHandler(console)

    # ── File Handler ──
    # Ensure logs directory exists
    log_dir = os.path.dirname(config.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        file_handler = logging.FileHandler(
            config.LOG_FILE,
            mode="a",           # append (don't overwrite old logs)
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # file always gets everything
        file_handler.setFormatter(SynFileFormatter())
        root.addHandler(file_handler)
    except OSError as e:
        # If we can't create the log file, just log to console
        console.setLevel(logging.DEBUG)
        root.warning(f"Could not create log file: {e}")

    # Enable ANSI colors on Windows
    _enable_windows_ansi()


def _enable_windows_ansi():
    """
    Enable ANSI color codes on Windows terminal.

    WHY THIS IS NEEDED:
        Windows cmd/PowerShell historically didn't support ANSI colors.
        Modern Windows 10+ does, but it needs to be enabled via the
        Windows API (SetConsoleMode with ENABLE_VIRTUAL_TERMINAL_PROCESSING).
    """
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences for stdout
            kernel32.SetConsoleMode(
                kernel32.GetStdHandle(-11),  # STD_OUTPUT_HANDLE
                7  # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL | ENABLE_VIRTUAL_TERMINAL
            )
        except Exception:
            pass  # fail silently — colors just won't work


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a SYN module.

    Usage:
        logger = get_logger("CLAP")    → creates logger "SYN.CLAP"
        logger = get_logger("STT")     → creates logger "SYN.STT"
        logger = get_logger("MAIN")    → creates logger "SYN.MAIN"

    HOW LOGGER HIERARCHY WORKS:
        All our loggers are children of "SYN":
            SYN          (root — has the handlers)
            ├── SYN.CLAP
            ├── SYN.STT
            ├── SYN.REC
            └── SYN.MAIN

        Child loggers inherit their parent's handlers.
        So SYN.CLAP automatically logs to both console AND file
        because its parent SYN has both handlers.

    Args:
        name: Short module name (e.g., "CLAP", "STT", "REC")

    Returns:
        A configured logging.Logger instance
    """
    _setup_root_logger()
    return logging.getLogger(f"SYN.{name}")
