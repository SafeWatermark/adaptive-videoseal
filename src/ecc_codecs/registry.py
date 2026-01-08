from ecc_codecs.bch_codec import BCHWatermarkCodec
from ecc_codecs.conv_codec import ConvWatermarkCodec
from ecc_codecs.hamming_codec import HammingWatermarkCodec
from ecc_codecs.polar_codec import PolarWatermarkCodec
from ecc_codecs.rs_codec import RSWatermarkCodec

ecc_codec_registry = {
    "bch": BCHWatermarkCodec,
    "conv": ConvWatermarkCodec,
    "hamming": HammingWatermarkCodec,
    "polar": PolarWatermarkCodec,
    "rs": RSWatermarkCodec,
    "default": PolarWatermarkCodec,
}
