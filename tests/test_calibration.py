"""Pitch calibration, against a synthetic pitch whose homography is known exactly.

Rendering a pitch through a homography we choose means there is a ground truth to compare
against, which real footage cannot give. The acceptance test on the real clip lives in
`test_acceptance_real_clip.py`.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibrate_pitch import (
    CIRCLE_RADIUS_M,
    calibrate,
    marking_mask,
    pitch_mask,
    to_world,
    widest_quad,
)

WIDTH_M = 68.0
GRASS = (60, 140, 70)
STANDS = (80, 80, 80)
PAINT = (250, 250, 250)
SIZE = (1280, 720)


def broadcast_homography() -> np.ndarray:
    """World metres -> image pixels, shaped like a camera on the halfway line."""
    world = np.float32([[-40, 0], [40, 0], [40, WIDTH_M], [-40, WIDTH_M]])
    image = np.float32([[300, 150], [980, 150], [1270, 680], [10, 680]])
    return cv2.getPerspectiveTransform(world, image)


#: Grass beyond the touchlines. Real pitches have run-off, and it matters here: paint drawn
#: hard against the stands is not surrounded by grass, so the pitch mask excludes it and the
#: touchlines become undetectable. The far side gets only a sliver, as it does in a stadium
#: where the advertising boards crowd the line -- which is why the far touchline is seeded
#: from the grass boundary and then pulled onto the paint.
FAR_RUNOFF_M = 0.6
NEAR_RUNOFF_M = 6.0


def render(homography: np.ndarray, occlusions: int = 0) -> np.ndarray:
    frame = np.full((SIZE[1], SIZE[0], 3), STANDS, np.uint8)

    def project(points):
        return to_world(homography, np.asarray(points, float)).astype(np.int32)

    # Extend well past the frame edges: a real pitch runs out of shot sideways, so the grass
    # boundary a column-wise scan sees is the touchline everywhere, not the slanted end of a
    # polygon.
    grass = [[-150, -FAR_RUNOFF_M], [150, -FAR_RUNOFF_M],
             [150, WIDTH_M + NEAR_RUNOFF_M], [-150, WIDTH_M + NEAR_RUNOFF_M]]
    cv2.fillPoly(frame, [project(grass)], GRASS)
    for y in (0.0, WIDTH_M):
        cv2.polylines(frame, [project([[x, y] for x in np.arange(-150, 150, 0.5)])], False, PAINT, 3)
    cv2.polylines(frame, [project([[0.0, y] for y in np.arange(0, WIDTH_M + 0.5, 0.5)])], False, PAINT, 3)
    theta = np.linspace(0, 2 * np.pi, 721)
    circle = np.column_stack([CIRCLE_RADIUS_M * np.cos(theta),
                              WIDTH_M / 2 + CIRCLE_RADIUS_M * np.sin(theta)])
    cv2.polylines(frame, [project(circle)], True, PAINT, 3)

    rng = np.random.default_rng(3)
    for _ in range(occlusions):
        x, y = rng.uniform(-20, 20), rng.uniform(10, 58)
        p = project([[x, y]])[0]
        cv2.rectangle(frame, (p[0] - 9, p[1] - 45), (p[0] + 9, p[1]), (30, 30, 200), -1)
    return frame


@pytest.fixture(scope="module")
def truth():
    return broadcast_homography()


@pytest.fixture(scope="module")
def frame(truth):
    return render(truth)


# -- the detection stages ----------------------------------------------------------------


def test_pitch_mask_finds_the_grass(frame):
    mask = pitch_mask(frame)
    assert 0.3 < (mask > 0).mean() < 0.95


def test_marking_mask_finds_paint_only_on_the_pitch(frame):
    mask = marking_mask(frame, pitch_mask(frame))
    assert 500 < (mask > 0).sum() < 0.2 * mask.size


def test_rejects_a_frame_with_no_pitch():
    with pytest.raises(RuntimeError):
        calibrate(np.full((480, 640, 3), STANDS, np.uint8), WIDTH_M, verbose=False)


# -- recovering the homography -------------------------------------------------------------


def world_error(recovered, truth, samples):
    """Round-trip world -> image (truth) -> world (recovered), in metres."""
    image_points = to_world(truth, samples)
    return np.linalg.norm(to_world(recovered, image_points) - samples, axis=1)


@pytest.fixture(scope="module")
def recovered(frame):
    homography, rms, _, _ = calibrate(frame, WIDTH_M, verbose=False)
    return homography, rms


def test_residuals_are_sub_metre(recovered):
    _, rms = recovered
    for name, value in rms.items():
        assert value < 1.0, f"{name}: {value:.2f} m"


def test_recovers_world_positions(recovered, truth):
    homography, _ = recovered
    samples = np.array([[x, y] for x in (-20, -10, 0, 10, 20) for y in (5, 20, 34, 48, 63)], float)
    error = world_error(homography, truth, samples)
    assert error.max() < 1.5, f"worst {error.max():.2f} m"
    assert np.median(error) < 0.5


def test_recovers_a_known_distance(recovered, truth):
    """The property that actually matters: a real distance must come back correct."""
    a, b = np.array([[-15.0, 20.0]]), np.array([[15.0, 20.0]])  # exactly 30 m apart
    homography, _ = recovered
    pa = to_world(homography, to_world(truth, a))[0]
    pb = to_world(homography, to_world(truth, b))[0]
    assert np.linalg.norm(pb - pa) == pytest.approx(30.0, abs=1.0)


def test_world_x_follows_image_x(recovered, frame):
    """The fit is symmetric about the halfway line, so the sign of x must be pinned.

    Every constraint -- halfway line at x = 0, touchlines parallel to it, circle straddling
    it -- is unchanged by x -> -x, so without a convention the optimiser returns one of two
    mirrored solutions depending on the arbitrary sign of the fitted ellipse's axis.
    Distances are the same either way, but the output should be reproducible.
    """
    homography, _ = recovered
    height, width = frame.shape[:2]
    probe = to_world(homography, np.array([[0.25 * width, 0.5 * height],
                                           [0.75 * width, 0.5 * height]]))
    assert probe[1, 0] > probe[0, 0]


def test_calibration_is_deterministic(frame):
    a, _, _, _ = calibrate(frame, WIDTH_M, verbose=False)
    b, _, _, _ = calibrate(frame, WIDTH_M, verbose=False)
    assert np.allclose(a / a[2, 2], b / b[2, 2], atol=1e-6)


def test_survives_players_occluding_the_markings(truth):
    """The centre circle arrives in fragments on real footage; RANSAC has to cope."""
    homography, _, _, _ = calibrate(render(truth, occlusions=14), WIDTH_M, verbose=False)
    samples = np.array([[x, y] for x in (-15, 0, 15) for y in (10, 34, 58)], float)
    assert world_error(homography, truth, samples).max() < 2.0


def test_width_scales_the_result(truth):
    """A narrower pitch means every recovered distance shrinks in proportion."""
    narrow = 64.0
    homography, _, _, _ = calibrate(render(truth), narrow, verbose=False)
    a = to_world(homography, to_world(truth, np.array([[0.0, 0.0]])))[0]
    b = to_world(homography, to_world(truth, np.array([[0.0, WIDTH_M]])))[0]
    assert np.linalg.norm(b - a) == pytest.approx(narrow, abs=1.5)


# -- the quad ------------------------------------------------------------------------------


def test_quad_is_convex_and_inside_the_frame(recovered, frame):
    homography, _ = recovered
    quad, length_m = widest_quad(homography, WIDTH_M, frame.shape)

    assert length_m > 10
    assert (quad >= 0).all() and (quad[:, 0] <= SIZE[0]).all() and (quad[:, 1] <= SIZE[1]).all()
    assert cv2.isContourConvex(quad.astype(np.float32))


def test_quad_feeds_the_view_transformer(recovered, frame):
    """The output has to be usable by the thing that consumes it."""
    from config import PitchConfig
    from view_transformer import ViewTransformer

    homography, _ = recovered
    quad, length_m = widest_quad(homography, WIDTH_M, frame.shape)
    view = ViewTransformer(PitchConfig(
        pixel_vertices=quad.tolist(), length_m=length_m, width_m=WIDTH_M, calibrated=True
    ))
    centre = view.transform_point(quad.mean(axis=0))
    assert centre is not None
    assert 0 <= centre[0] <= length_m and 0 <= centre[1] <= WIDTH_M
