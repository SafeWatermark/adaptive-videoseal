from scene_detectors.content_detector import ContentDetector
from scene_detectors.hash_detector import HashDetector
from scene_detectors.histogram_detector import HistogramDetector
from scene_detectors.lpips_detector import LpipsDetector

detector_registry = {
    "histogram": HistogramDetector,
    "content": ContentDetector,
    "hash": HashDetector,
    "lpips": LpipsDetector,
    "default": ContentDetector,
}
