import logging
from torch.nn import functional as F
import torch
from models.stega_stamp import ONNXStegaStampEncoder
from videoseal.models.embedder import Embedder
from models.stega_stamp import STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH


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
