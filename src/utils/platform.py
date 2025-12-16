"""Cross-platform utilities for Windows/Linux compatibility."""

import sys
import os

# Detect platform
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

# Platform name for CLI/logging
PLATFORM_NAME = "windows" if IS_WINDOWS else ("linux" if IS_LINUX else "macos")


def get_matplotlib_backend():
    """Return appropriate matplotlib backend for current platform."""
    if IS_WINDOWS:
        # TkAgg can work on Windows but Agg is safer for headless/scripts
        # Check if running in interactive mode
        if os.environ.get("DISPLAY") or os.environ.get("TERM_PROGRAM"):
            return "TkAgg"
        return "Agg"
    elif IS_MAC:
        return "MacOSX"
    else:
        # Linux - TkAgg if display available, else Agg
        if os.environ.get("DISPLAY"):
            return "TkAgg"
        return "Agg"


def configure_matplotlib():
    """Configure matplotlib for current platform. Call before importing pyplot."""
    import matplotlib
    backend = get_matplotlib_backend()
    try:
        matplotlib.use(backend)
    except Exception:
        # Fallback to Agg if preferred backend fails
        matplotlib.use("Agg")


def get_multiprocessing_context():
    """Return appropriate multiprocessing context for current platform."""
    import multiprocessing
    if IS_WINDOWS:
        # Windows requires 'spawn' for ProcessPoolExecutor
        return multiprocessing.get_context("spawn")
    else:
        # Linux/Mac can use 'fork' (faster) or 'spawn'
        return multiprocessing.get_context("fork")


def safe_cpu_count():
    """Return safe CPU count for parallel processing."""
    count = os.cpu_count()
    if count is None:
        return 1
    # Leave one core free
    return max(1, count - 1)


