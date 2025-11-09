import time
from functools import wraps
import logging


def log_elapsed_time():
    """Decorator to log elapsed time for a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"[TIMER] {func.__name__} finished in {elapsed:.2f} sec.")
            return result

        return wrapper

    return decorator
