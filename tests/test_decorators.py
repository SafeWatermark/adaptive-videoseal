import shutil
import subprocess

from src.decorators import patch_ffmpeg_decorator


def test_patch_ffmpeg_decorator(monkeypatch):
    """
    Test that patch_ffmpeg_decorator():
    - wraps the function
    - activates patch_ffmpeg_path during the wrapped function call
    - rewrites ffmpeg path inside the call
    - restores subprocess.run after the call
    """

    real_ffmpeg = shutil.which("ffmpeg")
    assert real_ffmpeg is not None

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return 0

    # Use monkeypatch so the test doesn't modify the real environment
    monkeypatch.setattr(subprocess, "run", fake_run)

    subprocess_args = ["-i", "input.mp4"]

    # Create a dummy function and decorate it
    @patch_ffmpeg_decorator
    def dummy_function():
        # Call ffmpeg with the bad path INSIDE the decorated function
        bad_path = "/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg"
        subprocess.run([bad_path, *subprocess_args])

    dummy_function()

    assert (
        calls[0][0] == real_ffmpeg
    ), f"Expected real ffmpeg '{real_ffmpeg}', but got '{calls[0][0]}'"

    # Assert all other arguments remain the same
    assert (
        calls[0][1:] == subprocess_args
    ), f"Unexpected args: {calls[0][1:]}, expected {subprocess_args}"


def test_patch_ffmpeg_decorator_restores_run_after_call():
    """
    Ensure the decorator itself restores subprocess.run and does not leak patches
    when not using monkeypatch.
    """

    original_run = subprocess.run

    @patch_ffmpeg_decorator
    def dummy():
        assert subprocess.run is not original_run  # must be patched inside

    dummy()

    assert subprocess.run is original_run  # must be restored after
