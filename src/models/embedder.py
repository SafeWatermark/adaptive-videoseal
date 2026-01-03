import logging
from torch.nn import functional as F
import torch
from models.stega_stamp import ONNXStegaStampEncoder
from videoseal.models.embedder import Embedder
from models.stega_stamp import STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH
from models.revmark import (
    REVMARK_DEFAULT_HEIGHT,
    REVMARK_DEFAULT_WIDTH,
    REVMARK_DEFAULT_MSG_BITS,
    REVMARK_RESIDUAL_SCALE,
    REVMARK_FRAME_SIZE,
)


class StegaStampEmbedder(Embedder):
    TARGET_SIZE = (STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH)

    def __init__(
        self,
        onnx_path: str,
    ) -> None:
        super(StegaStampEmbedder, self).__init__()
        self.stega_stamp_encoder = ONNXStegaStampEncoder(onnx_path)
        self.yuv = False

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        nbits = self.stega_stamp_encoder.num_bits
        return torch.randint(0, 2, (bsz, nbits))

    def _preprocess(self, imgs: torch.Tensor):
        """
        imgs: BCHW in [0,1]

        Returns:
            imgs_for_onnx: BHWC resized to 400×400 and normalized
            meta: dict storing original size, etc.
        """
        H, W = imgs.shape[-2:]
        meta = {"orig_size": (H, W)}

        if (H, W) != self.TARGET_SIZE:
            logging.warning(
                f"Expected image size {self.TARGET_SIZE}, but got {imgs.shape[-2:]}. "
                "Adjust the size by setting the `img_size(int)` class variable in VideoSeal."
            )

            # Resize BCHW → BCHW(400×400)
            imgs = F.interpolate(
                imgs,
                size=self.TARGET_SIZE,
                mode="bilinear",
                align_corners=False,
            )

        # BCHW → BHWC
        imgs_nhwc = imgs.permute(0, 2, 3, 1)

        return imgs_nhwc, meta

    def _postprocess(self, imgs_w: torch.Tensor, meta: dict):
        """
        imgs_w: BHWC output from ONNX model.
        meta: contains original H,W

        Converts:
            BHWC(400×400) → BCHW(original_size)
        Also applies inverse normalization.
        """
        H_orig, W_orig = meta["orig_size"]

        # NHWC → NCHW
        imgs_w = imgs_w.permute(0, 3, 1, 2)

        # Resize back to original size
        if (H_orig, W_orig) != self.TARGET_SIZE:
            imgs_w = F.interpolate(
                imgs_w, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )

        return imgs_w

    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Notes:
            - `blender.scaling_i` and `blender.scaling_w` must both be set to `1.0` in Videoseal.
            StegaStamp is designed for a 1:1 blend between the image and the watermark residual.
            - For the same reason, disable attenuation by setting:
                   model.attenuation = None
                Be aware that Videoseal 1.0 automatically enables attenuation, so make sure
                to override it if necessary.
            - Stega stamp satisfies this:
                torch.clamp(image + residual, 0, 1) == encoded_image
            - If `scaling_w` is smaller than `scaling_i`, watermark detection
            performance degrades significantly.
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW [0, 1]
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs, meta = self._preprocess(imgs)
        _, residual = self.stega_stamp_encoder(imgs, msgs)
        imgs_w = self._postprocess(residual, meta)
        return imgs_w


class REVMarkEmbedder(Embedder):
    """
    Embedder wrapper for REVMark videoseal model.

    Pre-trained REVMark operates on fixed 8-frame clips with shape (B, C, 8, H, W) and
    pixel values in [-1, 1]. This embedder handles:
    - Temporal chunking into 8-frame blocks
    - Temporal padding for incomplete blocks using nearest-neighbor upsampling
    - Optional perturbation smoothing for padded frames
    - Pixel range conversion between [0,1] and [-1,1]
    - Automatic resizing to match REVMark's expected resolution

    Note:
        When using this embedder with VideoSeal, configure the blender with
        scaling_w=0.5 to account for the domain conversion from [-1,1] to [0,1].
    """

    TARGET_SIZE = (REVMARK_DEFAULT_HEIGHT, REVMARK_DEFAULT_WIDTH)

    def __init__(
        self,
        encoder_path: str,
        msg_bits: int = REVMARK_DEFAULT_MSG_BITS,
        perturbation_epsilon: float = 0.0,
        watermark_mode: str = "per_block",
    ):
        """
        Initialize REVMarkEmbedder.

        Args:
            encoder_path: Path to the encoder checkpoint file (Encoder.pth).
            msg_bits: Number of message bits (default: 96).
            video_shape: [F, H, W] shape expected by encoder (default: [8, 128, 128]).
            perturbation_epsilon: Small noise magnitude for padded frames.
                                  Set to 0 to disable. Recommended: 0.002-0.01.
            wattermark_mode: Watermark insertion strategy.
                Options:
                    - "per_frame":
                        Inserts a message into every frame.
                        Higher accuracy; slower. All input messages are used.
                    - "per_block":
                        Inserts a single message per 8-frame block.
                        Faster; only the first message in each block is used.
        """
        super().__init__()

        self.video_shape = [
            REVMARK_FRAME_SIZE,
            REVMARK_DEFAULT_HEIGHT,
            REVMARK_DEFAULT_WIDTH,
        ]
        self.target_size = (self.video_shape[1], self.video_shape[2])
        self.msg_bits = msg_bits
        self.perturbation_epsilon = perturbation_epsilon
        self.yuv = False  # REVMark works on RGB
        self.watermark_mode = watermark_mode

        # Import REVMark components
        from REVMark.REVMark import Encoder, framenorm

        self.encoder = Encoder(msg_bits, self.video_shape)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        self.encoder.eval()
        self.encoder.tasblock.enable = True
        self._framenorm = framenorm

    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        """Generate random message bits."""
        return torch.randint(0, 2, (bsz, self.msg_bits))

    def _temporal_pad(self, frames: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """
        Pad frames to 8 using nearest-neighbor temporal upsampling.

        Uses deterministic mapping: padded[i] = original[floor(i * T / 8)]

        Examples:
            T=1 -> [0,0,0,0,0,0,0,0]
            T=2 -> [0,0,0,0,1,1,1,1]
            T=4 -> [0,0,1,1,2,2,3,3]
            T=7 -> [0,1,1,2,3,4,5,6]

        Args:
            frames: (T, C, H, W) tensor with T < 8 frames.

        Returns:
            padded: (8, C, H, W) tensor with padded frames.
            original_positions: List of positions in padded array that hold
                               the first occurrence of each original frame.
        """
        T = frames.shape[0]

        # Compute mapping: padded[i] = original[floor(i * T / 8)]
        indices = [int(i * T / REVMARK_FRAME_SIZE) for i in range(REVMARK_FRAME_SIZE)]
        padded = frames[indices].clone()

        # Track which positions hold original frames (first occurrence of each index)
        original_positions = []
        seen: set[int] = set()
        for i, idx in enumerate(indices):
            if idx not in seen:
                seen.add(idx)
                original_positions.append(i)

        return padded, original_positions

    def _add_perturbation_into_non_original_frames(
        self, padded: torch.Tensor, original_positions: list[int]
    ) -> torch.Tensor:
        # Apply perturbation to duplicate (non-original) frames
        original_set = set(original_positions)
        for i in range(REVMARK_FRAME_SIZE):
            if i not in original_set:
                noise = torch.randn_like(padded[i]) * self.perturbation_epsilon
                padded[i] = padded[i] + noise
        return padded

    def _preprocess(self, imgs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Preprocess frames for REVMark encoder.

        Args:
            imgs: (F, C, H, W) frames in [0, 1].

        Returns:
            imgs_processed: Resized frames in [-1, 1].
            meta: Dict with original size info.
        """
        H, W = imgs.shape[-2:]
        meta = {"orig_size": (H, W)}

        if (H, W) != self.target_size:
            logging.warning(
                f"Expected image size {self.target_size}, but got {imgs.shape[-2:]}. "
                "Adjust the size by setting the `img_size(int)` class variable in VideoSeal."
            )
            imgs = F.interpolate(
                imgs,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Convert [0, 1] -> [-1, 1]
        imgs = imgs * 2 - 1

        return imgs, meta

    def _postprocess(self, residual: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        Postprocess residual from REVMark encoder.

        Args:
            residual: (F, C, H, W) residual tensor.
            meta: Dict with original size info.

        Returns:
            Residual resized to original dimensions.
        """
        H_orig, W_orig = meta["orig_size"]

        if (H_orig, W_orig) != self.target_size:
            residual = F.interpolate(
                residual,
                size=(H_orig, W_orig),
                mode="bilinear",
                align_corners=False,
            )

        return residual

    def _encode_per_frame(
        self, padded_frames: torch.Tensor, padded_msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-frame watermark mode: encoder forward pass for each frame

        padded_frames: (B, C, H, W)
        padded_msgs:   (B, num_bits)

        Returns:
            r: (B, C, H, W)
        """
        device = padded_frames.device
        B = padded_frames.shape[0]

        # Construct circular-shifted temporal stacks
        idx = (
            torch.arange(B, device=device).unsqueeze(1)
            + torch.arange(B, device=device).unsqueeze(0)
        ) % B

        padded_batches = padded_frames[idx]  # (B, B, C, H, W)
        encoder_input = padded_batches.permute(0, 2, 1, 3, 4)  # (B, C, B, H, W)

        # Per-frame messages
        r = self.encoder(encoder_input, padded_msgs)  # (B, C, B, H, W)

        # Apply framenorm and extract temporal index 0
        r = REVMARK_RESIDUAL_SCALE * self._framenorm(r)
        return r[:, :, 0, :, :]  # (B, C, H, W)

    def _encode_per_block(
        self, padded_frames: torch.Tensor, padded_msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-block watermark mode: encoder forward pass for the entire block.

        padded_frames: (B, C, H, W)
        padded_msgs:   (B, num_bits)

        Returns:
            r: (B, C, H, W)
        """
        # Convert to (1, C, B, H, W)
        encoder_input = padded_frames.permute(1, 0, 2, 3).unsqueeze(0)

        # One message for the entire block — use the first
        block_msg = padded_msgs[0].unsqueeze(0)  # (1, num_bits)

        r = self.encoder(encoder_input, block_msg)  # (1, C, B, H, W)
        r = REVMARK_RESIDUAL_SCALE * self._framenorm(r)

        return r.squeeze(0).permute(1, 0, 2, 3)  # → (B, C, H, W)

    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Compute watermark residual for input frames.

        The residual is scaled by 6.2 * framenorm() as per REVMark's design.

        Args:
            imgs: (F, C, H, W) frames in [0, 1].
            msgs: (B, num_bits) message tensor.

        Returns:
            residual: (F, C, H, W) watermark residual.
        """

        F = imgs.shape[0]

        # Preprocess
        imgs_processed, meta = self._preprocess(imgs)

        # Process in 8-frame chunks
        residuals = []

        for start in range(0, F, REVMARK_FRAME_SIZE):
            end = min(start + REVMARK_FRAME_SIZE, F)
            chunk = imgs_processed[start:end]
            chunk_len = end - start

            # Pad if needed
            if chunk_len < REVMARK_FRAME_SIZE:
                padded_frames, original_positions = self._temporal_pad(chunk)
                if self.perturbation_epsilon > 0:
                    padded_frames = self._add_perturbation_into_non_original_frames(
                        padded_frames, original_positions
                    )
                padded_msgs, _ = self._temporal_pad(msgs[start:end])
            else:
                padded_frames = chunk
                padded_msgs = msgs[start:end]
                original_positions = list(range(REVMARK_FRAME_SIZE))

            if self.watermark_mode == "per_frame":
                r = self._encode_per_frame(padded_frames, padded_msgs)
            # TODO: we can boost speed by batch encoding(delete for statement)
            elif self.watermark_mode == "per_block":
                r = self._encode_per_block(padded_frames, padded_msgs)
            else:
                raise ValueError(f"Invalid watermark_mode: {self.watermark_mode}")

            # Extract residuals for original frames only
            if chunk_len < REVMARK_FRAME_SIZE:
                r = r[original_positions]

            residuals.append(r)

        residual = torch.cat(residuals, dim=0)

        # Postprocess (resize back if needed)
        residual = self._postprocess(residual, meta)

        return residual
