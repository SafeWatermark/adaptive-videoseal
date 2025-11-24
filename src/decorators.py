import time
import functools
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


def make_interceptor(before=None, after=None):
    """
    Creates a decorator-like wrapper WITHOUT modifying the source code.

    before(func, args, kwargs) -> optionally modify args/kwargs
    after(func, args, kwargs, result) -> optionally modify result
    """

    def wrap_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- BEFORE CALL ---
            if before is not None:
                new = before(func, args, kwargs)
                if new is not None:
                    args, kwargs = new  # allow mutation

            # --- CALL ORIGINAL FUNCTION ---
            result = func(*args, **kwargs)

            # --- AFTER CALL ---
            if after is not None:
                new_result = after(func, args, kwargs, result)
                if new_result is not None:
                    result = new_result  # allow mutation

            return result

        return wrapper

    return wrap_function


def make_video_mode_recorder():
    records = []

    def on_apply_video_mode_start(func, args, kwargs):
        if func.__name__ != "_apply_video_mode":
            raise ValueError("this callback is for _apply_video_mode")
        preds_w, total_frames, _, _ = args
        t = preds_w.size(0)
        records.append(
            {
                "n_embed": t,
                "embed_ratio": t / total_frames,
                "total_frames": total_frames,
            }
        )
        return None

    return on_apply_video_mode_start, records
