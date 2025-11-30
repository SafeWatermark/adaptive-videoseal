import subprocess
import shutil

from src.utils import patch_ffmpeg_path


def test_patch_ffmpeg_path(monkeypatch):
    """
    Test that patch_ffmpeg_path() correctly rewrites calls to subprocess.run
    where the command begins with the specified 'bad' ffmpeg path.

    Expectations:
    - The bad ffmpeg path is replaced with the real system ffmpeg path.
    - All additional command arguments remain unchanged.
    """
    real_ffmpeg = shutil.which("ffmpeg")
    assert real_ffmpeg is not None

    # Track what command subprocess.run receives
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return 0  # pretend success

    # replace real subprocess.run temporarily
    monkeypatch.setattr(subprocess, "run", fake_run)

    bad_path = "/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg"
    subprocess_args = ["-i", "input.mp4"]

    with patch_ffmpeg_path(bad_path=bad_path):
        # simulate a user calling ffmpeg with the bad path
        subprocess.run([bad_path, *subprocess_args])

    # ensure the command was rewritten to the real ffmpeg path
    assert (
        calls[0][0] == real_ffmpeg
    ), f"Expected real ffmpeg '{real_ffmpeg}', but got '{calls[0][0]}'"

    # Assert all other arguments remain the same
    assert (
        calls[0][1:] == subprocess_args
    ), f"Unexpected args: {calls[0][1:]}, expected {subprocess_args}"


def test_subprocess_restoration_after_patch_ffmpeg():
    """
    Test that patch_ffmpeg_path() restores subprocess.run
    to its original function after exiting the context manager.
    """
    original_run = subprocess.run

    with patch_ffmpeg_path():
        assert (
            subprocess.run is not original_run
        ), "subprocess.run should be patched inside the context"

    assert (
        subprocess.run is original_run
    ), "subprocess.run should be restored after exiting the context"
