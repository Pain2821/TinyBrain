# core/utils_logging.py
import sys
import time

def log(*args, **kwargs):
    """Simple log wrapper to centralize printing and toggling verbosity."""
    print(*args, **kwargs)

def vlog(enabled, *args, **kwargs):
    if enabled:
        log(*args, **kwargs)

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
