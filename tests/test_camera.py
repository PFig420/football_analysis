"""Camera model acceptance tests.

Two models, one contract: given frames, produce a per-frame homography into the reference
view. A synthetic pan has a known answer, so there is no ambiguity about what correct means.

The original estimator fails every test in here. Measured against this suite it reported a
final drift of zero on a 48 px pan, and 22.5 px of camera motion for a camera that never
moved.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from camera_movement_estimator import (
    CameraMovementEstimator,
    CameraModel,
    HomographyCameraModel,
    make_camera_model,
)
from config import CameraConfig

#: Detect at full resolution: the synthetic fixtures are small to begin with.
FULL_RES = dict(detector_scale=1.0)


def translation_model() -> CameraMovementEstimator:
    return CameraMovementEstimator(CameraConfig(model="translation"))


def homography_model() -> HomographyCameraModel:
    return HomographyCameraModel(CameraConfig(model="homography", **FULL_RES))


BOTH_MODELS = [pytest.param(translation_model, id="translation"),
               pytest.param(homography_model, id="homography")]


def shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate an image by a sub-pixel amount."""
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


def pan_sequence(image: np.ndarray, per_frame: tuple[float, float], n: int) -> list[np.ndarray]:
    """A clip in which the scene moves by `per_frame` every frame."""
    return [shift(image, per_frame[0] * i, per_frame[1] * i) for i in range(n)]


# -- both models -------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", BOTH_MODELS)
@pytest.mark.parametrize("per_frame", [(2.0, 0.0), (0.0, -1.5), (1.25, 0.75), (-3.0, 2.0)])
def test_recovers_known_pan(textured_image, make_model, per_frame):
    frames = pan_sequence(textured_image, per_frame, 25)
    movement = make_model().estimate_sequence(frames)

    expected = np.array([[per_frame[0] * i, per_frame[1] * i] for i in range(len(frames))])
    assert movement.shape == expected.shape
    assert np.abs(movement - expected).max() < 1.0


@pytest.mark.parametrize("make_model", BOTH_MODELS)
def test_output_is_cumulative_not_per_segment(textured_image, make_model):
    """The consumer maps positions back to frame 0, so the mapping must accumulate."""
    frames = pan_sequence(textured_image, (2.0, 0.0), 30)
    movement = make_model().estimate_sequence(frames)

    x = movement[:, 0]
    assert np.all(np.diff(x) > 0), "a steady one-way pan must produce monotonic drift"
    # The old estimator returned to zero whenever motion dipped below its threshold.
    assert x[-1] == pytest.approx(2.0 * 29, abs=1.0)


@pytest.mark.parametrize("make_model", BOTH_MODELS)
def test_static_camera_reports_no_drift(textured_image, make_model):
    frames = [textured_image.copy() for _ in range(20)]
    movement = make_model().estimate_sequence(frames)
    assert np.abs(movement).max() < 0.5, "a static camera must not accumulate phantom drift"


@pytest.mark.parametrize("make_model", BOTH_MODELS)
def test_moving_foreground_does_not_fool_the_model(textured_image, make_model):
    """A large object crossing a static scene must not be read as camera motion.

    This is the specific failure of the old max-displacement rule: one feature on a running
    player became the camera's motion.
    """
    frames = []
    for i in range(24):
        frame = textured_image.copy()
        x = 40 + 18 * i  # a bright block sweeping across, faster than any real pan
        cv2.rectangle(frame, (x, 150), (x + 130, 330), 255, -1)
        frames.append(frame)

    movement = make_model().estimate_sequence(frames)
    assert np.abs(movement).max() < 2.0, (
        f"foreground motion leaked into the camera estimate: {np.abs(movement).max():.1f} px"
    )


@pytest.mark.parametrize("make_model", BOTH_MODELS)
def test_update_matches_estimate_homographies(textured_image, make_model):
    frames = pan_sequence(textured_image, (1.0, 1.0), 10)

    batch = make_model().estimate_homographies(frames)

    incremental = make_model()
    streamed = np.array([incremental.update(f) for f in frames])

    assert np.allclose(batch, streamed)


@pytest.mark.parametrize("make_model", BOTH_MODELS)
def test_returns_three_by_three_homographies(textured_image, make_model):
    frames = pan_sequence(textured_image, (1.0, 0.0), 5)
    homographies = make_model().estimate_homographies(frames)

    assert homographies.shape == (5, 3, 3)
    assert np.allclose(homographies[0], np.eye(3)), "frame 0 is the reference"


# -- homography-specific -----------------------------------------------------------------


def test_homography_recovers_zoom(textured_image):
    """The case a translation model cannot represent at all.

    A camera zooming in scales the image about its centre. A translation fits this with an
    offset of zero and a large residual; a homography recovers the scale.
    """
    height, width = textured_image.shape[:2]
    centre = (width / 2, height / 2)

    frames, scales = [], []
    for i in range(12):
        factor = 1.0 + 0.012 * i
        matrix = cv2.getRotationMatrix2D(centre, 0.0, factor)
        frames.append(cv2.warpAffine(textured_image, matrix, (width, height),
                                     borderMode=cv2.BORDER_REFLECT))
        scales.append(factor)

    homographies = homography_model().estimate_homographies(frames)

    for homography, factor in zip(homographies, scales):
        # Mapping back to the reference should undo the zoom, so a point's offset from the
        # centre shrinks by exactly the factor it was grown by.
        probe = (centre[0] + 100.0, centre[1])
        q = homography @ np.array([probe[0], probe[1], 1.0])
        q = q[:2] / q[2]
        assert (q[0] - centre[0]) == pytest.approx(100.0 / factor, abs=2.0)


def test_homography_handles_a_pan_beyond_the_reference_view(textured_image):
    """A pan large enough to leave the keyframe behind must re-anchor, not fail."""
    wide = np.hstack([textured_image, textured_image[:, ::-1], textured_image])
    frames = [wide[:, x : x + 320] for x in range(0, 900, 12)]

    model = homography_model()
    model.estimate_homographies(frames)

    assert model.frames_without_estimate == 0
    assert model.keyframe_changes >= 1, "expected the keyframe to be re-anchored"


# -- shared machinery ---------------------------------------------------------------------


def test_add_adjust_positions_maps_into_the_reference_view():
    from camera_movement_estimator import translation_homography

    tracks = {"players": [{7: {"position": (100.0, 200.0)}}, {7: {"position": (110.0, 200.0)}}]}
    # The scene shifted 10 px right by frame 1; the player did not move.
    homographies = np.array([np.eye(3), translation_homography(10.0, 0.0)])

    CameraModel.add_adjust_positions_to_tracks(tracks, homographies)

    assert tracks["players"][0][7]["position_adjusted"] == pytest.approx((100.0, 200.0))
    assert tracks["players"][1][7]["position_adjusted"] == pytest.approx((100.0, 200.0))


def test_pan_at_measures_view_motion_at_the_centre():
    from camera_movement_estimator import translation_homography

    dx, dy = CameraModel.pan_at(translation_homography(30.0, -12.0), 1920, 1080)
    assert (dx, dy) == pytest.approx((30.0, -12.0))


def test_factory_selects_the_configured_model():
    assert isinstance(make_camera_model(CameraConfig(model="homography")), HomographyCameraModel)
    assert isinstance(make_camera_model(CameraConfig(model="translation")), CameraMovementEstimator)


def test_factory_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown camera model"):
        make_camera_model(CameraConfig(model="kalman-vibes"))


def test_default_model_is_the_homography():
    """A translation is wrong for broadcast footage; the default must not be the cheap one."""
    assert CameraConfig().model == "homography"
    assert isinstance(make_camera_model(), HomographyCameraModel)


def test_reset_clears_state(textured_image):
    model = homography_model()
    model.estimate_homographies(pan_sequence(textured_image, (3.0, 0.0), 10))
    model.reset()
    assert np.allclose(model.update(textured_image), np.eye(3))
    assert model.frames_seen == 1
