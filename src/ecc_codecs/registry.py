from ecc_codecs.conv_codec import ConvWatermarkCodec
from ecc_codecs.polar_codec import PolarWatermarkCodec
from ecc_codecs.rs_codec import RSWatermarkCodec

ecc_codec_registry = {
    "conv": ConvWatermarkCodec,
    "polar": PolarWatermarkCodec,
    "rs": RSWatermarkCodec,
    "default": PolarWatermarkCodec,
}
