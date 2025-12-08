import torch
import torch.nn as nn
import onnx2torch

STEGA_STAMP_IMAGE_HEIGHT = 400
STEGA_STAMP_IMAGE_WIDTH = 400
STEGA_STAMP_SCALING_W = 1.0
STEGA_STAMP_SCALING_I = 1.0
STEGA_STAMP_ATTENUATION = None
STEGA_STAMP_NUM_BITS = 100


class ONNXStegaStamp(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        self.onnx_path = onnx_path
        self.model = onnx2torch.convert(self.onnx_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ONNXStegaStampEncoder(ONNXStegaStamp):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.num_bits = STEGA_STAMP_NUM_BITS

    def forward(
        self, image: torch.Tensor, secret: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(secret, image)
        encoded_image, residual, _ = out
        return encoded_image, residual


class ONNXStegaStampDecoder(ONNXStegaStamp):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.num_bits = STEGA_STAMP_NUM_BITS

    def forward(self, watermark_image: torch.Tensor) -> torch.Tensor:
        dummy_secret = torch.zeros(
            (watermark_image.size(0), self.num_bits), device=watermark_image.device
        )
        _, _, decoded_secret_binary = self.model(dummy_secret, watermark_image)
        return decoded_secret_binary
