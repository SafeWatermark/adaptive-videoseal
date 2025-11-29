"""
MPEG4/H.264 compression perturbation.

Based on original VideoMarkBench implementation, adapted for in-memory processing.
"""

import torch
import numpy as np
import io
import av
import torch.nn as nn


class MPEG4(nn.Module):
    """
    MPEG4/H.264 compression perturbation.

    Adapted from original VideoMarkBench to work in-memory instead of file-based.
    """

    def __init__(self, quality, codec="libx264", fps=25):
        """
        Initialize MPEG4 compression perturbation.

        Args:
            quality: CRF quality value (0-51, higher = lower quality)
            codec: Codec to use ('libx264', 'libx265', etc.)
            fps: Frames per second
        """
        super(MPEG4, self).__init__()
        self.quality = quality
        self.codec = codec
        self.fps = fps

    def forward(self, video):
        """
        Apply MPEG4 compression to video.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Compressed video with same shape
        """
        T, C, H, W = video.shape

        # Convert to numpy (T, C, H, W) -> (T, H, W, C)
        frames = (video * 255).byte().cpu().numpy()
        frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C)

        # Use PyAV for compression (in-memory processing)
        buffer = io.BytesIO()

        try:
            # Write video to buffer
            with av.open(buffer, mode="w", format="mp4") as container:
                stream = container.add_stream(self.codec, rate=self.fps)
                stream.width = W
                stream.height = H
                stream.pix_fmt = "yuv420p" if self.codec != "libx264rgb" else "rgb24"
                stream.options = {
                    "crf": str(self.quality),
                    "preset": "ultrafast",  # Fast encoding
                    "tune": "zerolatency",  # Low latency
                }

                for frame_arr in frames:
                    frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            buffer.seek(0)

            # Read compressed video
            compressed_frames = []
            with av.open(buffer, mode="r") as container:
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="rgb24")
                    compressed_frames.append(img)

            # Convert back to tensor
            if len(compressed_frames) == 0:
                return video  # Fallback if compression fails

            compressed_array = np.stack(compressed_frames, axis=0)  # (T, H, W, C)
            compressed_array = np.transpose(
                compressed_array, (0, 3, 1, 2)
            )  # (T, C, H, W)
            compressed_tensor = torch.from_numpy(compressed_array).float() / 255.0

            # Ensure same number of frames
            if compressed_tensor.shape[0] != T:
                # Interpolate or pad
                if compressed_tensor.shape[0] < T:
                    # Repeat last frame
                    last_frame = compressed_tensor[-1:].repeat(
                        T - compressed_tensor.shape[0], 1, 1, 1
                    )
                    compressed_tensor = torch.cat(
                        [compressed_tensor, last_frame], dim=0
                    )
                else:
                    compressed_tensor = compressed_tensor[:T]

            return compressed_tensor.clamp(0.0, 1.0).to(video.device)

        except Exception:
            # Fallback: return original video if compression fails
            return video
