import os
import subprocess
import tempfile
import torch
import ffmpeg

from src.main import embed_video
from src.main import get_video_info


class FakeModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """A deterministic fake watermark model."""

    def get_random_msg(self):
        return torch.tensor([[0, 1, 0, 1]])

    def embed(self, imgs, *args, **kwargs):
        imgs_w = torch.zeros_like(imgs)
        msgs = torch.tensor([[0, 1, 0, 1]])

        outputs = {
            "imgs_w": imgs_w,  # watermarked imgs: f 3 h w
            "msgs": msgs[0:1].repeat(len(imgs), 1),  # original messages: f k
        }
        return outputs


def make_test_video(path, num_frames=10, width=64, height=48):
    """
    Creates a real MP4 file using ffmpeg and random torch frames.
    """
    frames = []
    for _ in range(num_frames):
        # random pixel data in uint8
        frame = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
        frames.append(frame.numpy())

    # FFmpeg raw video → MP4
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", r=30
        )
        .output(path, vcodec="libx264", pix_fmt="yuv420p", r=30, crf=23)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for f in frames:
        process.stdin.write(f.tobytes())

    process.stdin.close()
    process.wait()

    return frames


def count_frames(path):
    """Uses ffprobe to count frames."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return int(out)


# ------------------------------------------
# INTEGRATION TEST: leftover frame execution
# ------------------------------------------


def test_embed_video_leftover_real():
    chunk_size = 4
    num_frames = 10  # <-- leaves 2 leftover frames

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "input.mp4")
        output_path = os.path.join(tmp, "output.mp4")

        # create real test video
        make_test_video(input_path, num_frames=num_frames)

        # run real function
        model = FakeModel()
        embed_video(model, input_path, output_path, chunk_size=chunk_size)

        # verify output exists
        assert os.path.isfile(output_path)

        # verify frame count preserved
        output_frames = count_frames(output_path)
        assert (
            output_frames == num_frames
        ), f"Expected {num_frames} frames, got {output_frames}"

        # Extra check: ensure leftover chunk was processed
        info_input = get_video_info(input_path)
        info_output = get_video_info(output_path)

        assert info_input["num_frames"] == info_output["num_frames"]
