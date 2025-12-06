import logging
from videoseal.models.extractor import Extractor
import torch
from models.stega_stamp import ONNXStegaStampDecoder
from torch.nn import functional as F
from models.stega_stamp import STEGA_STAMP_IMAGE_HEIGHT, STEGA_STAMP_IMAGE_WIDTH


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
