"""Camera motion models.

`make_camera_model(config)` picks one from configuration. The default is the homography
model; see `base.py` for why a translation is not good enough on broadcast footage.
"""
from config import CameraConfig

from .base import CameraModel, apply_homography, translation_homography
from .camera_movement_estimator import CameraMovementEstimator
from .homography_model import HomographyCameraModel

MODELS = {
    "homography": HomographyCameraModel,
    "translation": CameraMovementEstimator,
}


def make_camera_model(config: CameraConfig | None = None) -> CameraModel:
    config = config or CameraConfig()
    try:
        return MODELS[config.model](config)
    except KeyError:
        raise ValueError(
            f"unknown camera model {config.model!r}; choose one of {sorted(MODELS)}"
        ) from None


__all__ = [
    "CameraModel",
    "CameraMovementEstimator",
    "HomographyCameraModel",
    "make_camera_model",
    "apply_homography",
    "translation_homography",
    "MODELS",
]
