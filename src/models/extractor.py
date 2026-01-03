import logging
from videoseal.models.extractor import Extractor
import torch
from models.stega_stamp import ONNXStegaStampDecoder
from torch.nn import functional as F
from models.stega_stamp import STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH
from models.revmark import (
    REVMARK_DEFAULT_HEIGHT,
    REVMARK_DEFAULT_WIDTH,
    REVMARK_DEFAULT_MSG_BITS,
    REVMARK_FRAME_SIZE,
)


class StegaStampExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    TARGET_SIZE = (STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH)

    def __init__(
        self,
        onnx_path: str,
    ) -> None:
        super(StegaStampExtractor, self).__init__()
        self.stega_stamp_decoder = ONNXStegaStampDecoder(onnx_path)

    def _preprocess(self, imgs: torch.Tensor):
        """
        imgs: BCHW in [0,1]

        Returns:
            imgs_for_onnx: BHWC resized to 400×400 and normalized
            meta: dict storing original size, etc.
        """
        # Resize BCHW → BCHW(400×400)
        if imgs.shape[-2:] != self.TARGET_SIZE:
            logging.warning(
                f"Expected image size {self.TARGET_SIZE}, but got {imgs.shape[-2:]}. "
                "Adjust the size by setting the `img_size(int)` class variable in VideoSeal."
            )
            imgs = F.interpolate(
                imgs,
                size=self.TARGET_SIZE,
                mode="bilinear",
                align_corners=False,
            )

        # BCHW → BHWC
        imgs_nhwc = imgs.permute(0, 2, 3, 1)

        return imgs_nhwc

    def _postprocess(self, masks: torch.Tensor) -> torch.Tensor:
        """
        masks: [B, nbits]

        Returns:
            [B, 1 + n_bits] by inserting a dummy probability.
        """
        B = masks.size(0)

        # Create dummy probability
        dummy_probability = torch.zeros(B, 1, device=masks.device, dtype=masks.dtype)

        return torch.cat([dummy_probability, masks], dim=1)

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW [0, 1]
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs_proc = self._preprocess(imgs)
        masks = self.stega_stamp_decoder(imgs_proc)
        masks = self._postprocess(masks)
        return masks


class REVMarkExtractor(Extractor):
    """
    Extractor wrapper for REVMark video watermarking model.

    REVMark decoder operates on fixed 8-frame clips with shape (B, C, 8, H, W) and
    pixel values in [-1, 1]. This extractor handles:
    - Temporal chunking into 8-frame blocks
    - Temporal padding for incomplete blocks using nearest-neighbor upsampling
    - Pixel range conversion between [0,1] and [-1,1]
    - Automatic resizing to match REVMark's expected resolution
    - Aggregation of decoded messages across chunks
    """

    TARGET_SIZE = (REVMARK_DEFAULT_HEIGHT, REVMARK_DEFAULT_WIDTH)

    def __init__(
        self,
        decoder_path: str,
        msg_bits: int = REVMARK_DEFAULT_MSG_BITS,
        detection_mode: str = "per_block",
    ):
        """
        Initialize REVMarkExtractor.

        Args:
            decoder_path: Path to the decoder checkpoint file (Decoder.pth).
            msg_bits: Number of message bits (default: 96).
            detection_mode:
                "per_block": (Efficiency-oriented)
                    Processes the video in non-overlapping consecutive 8-frame chunks.

                    Example:
                        [I_0, I_1, ..., I_7] -> [W_0, W_1, ..., W_7] (I: image, W: Watermark)

                    All watermarks(W_0, ..., W_7) have different values but have same message inserted

                "per_frame": (Precision-oriented)
                    Processes each frame as a 'first frame' by applying a circular
                    temporal shift to the 8-frame window.

                    It is recommended for high-precision extraction or scenarios
                    where different messages are embedded in adjacent frames.

                    Example:
                        [I_0, I_1, ..., I_7] -> [W_0, W_1, ..., W_7], Extract W_0 for I_0
                        [I_1, I_2, ..., I_0] -> [W_1, W_2, ..., W_0],  Extract W_1 for I_1
                        ...
                        [I_7, I_0, ..., I_6] -> [W_7, W_0, ..., W_6], Extract W_7 for I_7

                    All watermarks(W_0, ..., W_7) can have different messages

                Note:
                    It is highly recommended to use the same mode for both embedding
                    and extraction to ensure consistent message alignment and detection accuracy.
        """
        super().__init__()

        self.video_shape = [
            REVMARK_FRAME_SIZE,
            REVMARK_DEFAULT_HEIGHT,
            REVMARK_DEFAULT_WIDTH,
        ]
        self.target_size = (self.video_shape[1], self.video_shape[2])
        self.msg_bits = msg_bits
        self.detection_mode = detection_mode

        from REVMark.REVMark import Decoder

        self.decoder = Decoder(msg_bits, self.video_shape)
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location="cpu", weights_only=True)
        )
        self.decoder.eval()
        self.decoder.tasblock.enable = True

    def _temporal_pad(self, frames: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """
        Pad frames to 8 using nearest-neighbor temporal upsampling.

        Uses deterministic mapping: padded[i] = original[floor(i * T / 8)]

        Args:
            frames: (T, C, H, W) tensor with T < 8 frames.

        Returns:
            padded: (8, C, H, W) tensor with padded frames.
        """
        T = frames.shape[0]
        indices = [int(i * T / REVMARK_FRAME_SIZE) for i in range(REVMARK_FRAME_SIZE)]

        original_positions = []
        seen: set[int] = set()
        for i, idx in enumerate(indices):
            if idx not in seen:
                seen.add(idx)
                original_positions.append(i)

        return frames[indices].clone(), original_positions

    def _preprocess(self, imgs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Preprocess frames for REVMark decoder.

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

    def _postprocess(self, msgs: torch.Tensor) -> torch.Tensor:
        """
        Postprocess decoded messages to match VideoSeal format.

        Args:
            msgs: (B, nbits) decoded message probabilities.

        Returns:
            (B, 1 + nbits) tensor with dummy probability prepended.
        """
        B = msgs.size(0)
        dummy_probability = torch.zeros(B, 1, device=msgs.device, dtype=msgs.dtype)
        return torch.cat([dummy_probability, msgs], dim=1)

    def _decode_per_frame(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Decode messages per frame using circular temporal shifts.

        Args:
            chunk: (B, C, H, W)

        Returns:
            msgs: (B, num_bits)
        """
        device = chunk.device
        B = chunk.shape[0]

        # idx: (B, B) circular-shift indices
        idx = (
            torch.arange(B, device=device).unsqueeze(1)
            + torch.arange(B, device=device).unsqueeze(0)
        ) % B

        # (B, B, C, H, W)
        chunk_batches = chunk[idx]

        # (B, C, B, H, W)
        decoder_input = chunk_batches.permute(0, 2, 1, 3, 4)

        # Decode each frame
        msgs = self.decoder(decoder_input)  # (B, num_bits)

        return msgs

    def _decode_per_block(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Decode one message for the entire block.

        Args:
            chunk: (B, C, H, W)

        Returns:
            msgs: (B, num_bits) broadcasted to all frames
        """
        B = chunk.shape[0]

        # (1, C, B, H, W)
        decoder_input = chunk.permute(1, 0, 2, 3).unsqueeze(0)

        # Decode block
        msg = self.decoder(decoder_input)  # (1, num_bits)

        # Broadcast result across all frames
        return msg.repeat(B, 1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark message from input frames.

        Processes frames in 8-frame chunks and aggregates the decoded messages
        by averaging across all chunks.

        Args:
            imgs: (F, C, H, W) frames in [0, 1].

        Returns:
            msgs: (F, 1 + nbits) tensor with dummy probability and message bits.
        """
        F = imgs.shape[0]

        # Preprocess
        imgs_processed, _ = self._preprocess(imgs)

        # Process in 8-frame chunks
        decoded_msgs = torch.empty((F, self.msg_bits), device=imgs.device)
        for start in range(0, F, REVMARK_FRAME_SIZE):
            end = min(start + REVMARK_FRAME_SIZE, F)
            chunk = imgs_processed[start:end]
            chunk_len = end - start

            # Pad if needed
            if chunk_len < REVMARK_FRAME_SIZE:
                chunk, original_positions = self._temporal_pad(chunk)

            if self.detection_mode == "per_frame":
                msg_block = self._decode_per_frame(chunk)  # (8, num_bits)
            elif self.detection_mode == "per_block":
                msg_block = self._decode_per_block(chunk)  # (8, num_bits)
            else:
                raise ValueError(f"Invalid detection_mode: {self.detection_mode}")
            if chunk_len < REVMARK_FRAME_SIZE:
                msg_block = msg_block[original_positions]

            decoded_msgs[start:end, :] = msg_block

        return self._postprocess(decoded_msgs)
