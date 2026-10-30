"""Acceptance tests that need the real sample clip.

Skipped automatically when `input_videos/08fd33_4.mp4` is absent, so the rest of the suite
stays runnable with nothing but the repository. The clip is linked from the README.

These are slow -- registering 750 frames takes about half a minute. Deselect with
`pytest -m "not slow"`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_movement_estimator import make_camera_model
from config import CameraConfig
from utils import iter_frames, probe_video

VIDEO = Path(__file__).resolve().parent.parent / "input_videos" / "08fd33_4.mp4"
PROBE_FRAMES = [60, 150, 300, 450, 600, 749]

#: The criterion. A camera model that cannot place static scenery to within a pixel is
#: injecting error larger than the player movement being measured.
MAX_RESIDUAL_PX = 1.5

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not VIDEO.exists(), reason=f"{VIDEO.name} not present; see the README"),
]


@pytest.fixture(scope="module")
def clip_meta():
    return probe_video(VIDEO)


@pytest.fixture(scope="module")
def registered():
    """Homographies from both models, plus the probe frames, computed in one pass."""
    models = {name: make_camera_model(CameraConfig(model=name))
              for name in ("homography", "translation")}
    homographies = {name: [] for name in models}
    frames = {}

    for i, frame in enumerate(iter_frames(VIDEO)):
        for name, model in models.items():
            homographies[name].append(model.update(frame))
        if i == 0 or i in PROBE_FRAMES:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return {name: np.array(v) for name, v in homographies.items()}, frames, models


def static_correspondences(frames, n):
    """Matched points between frame 0 and frame n, keeping only static scenery.

    Players move independently of the camera, so RANSAC rejects them as outliers; what
    survives is stadium structure and pitch markings, whose true motion is the camera's.
    """
    sift, matcher = cv2.SIFT_create(4000), cv2.BFMatcher(cv2.NORM_L2)
    k0, d0 = sift.detectAndCompute(frames[0], None)
    k1, d1 = sift.detectAndCompute(frames[n], None)

    good = [m for m, s in matcher.knnMatch(d1, d0, k=2) if m.distance < 0.75 * s.distance]
    assert len(good) >= 30, f"only {len(good)} matches to frame {n}"

    source = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([k0[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    keep = mask.ravel() == 1
    return source[keep].reshape(-1, 2), target[keep].reshape(-1, 2)


def residual(homography, source, target):
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return float(np.median(np.linalg.norm(projected - target, axis=1)))


# -- the clip itself ---------------------------------------------------------------------


def test_clip_is_25_fps_not_24(clip_meta):
    """The original hardcoded 24, so every speed it reported was 4% low on its own clip."""
    assert clip_meta.fps == pytest.approx(25.0, abs=0.01)
    assert (clip_meta.width, clip_meta.height) == (1920, 1080)
    assert clip_meta.frame_count == 750


# -- the camera model --------------------------------------------------------------------


@pytest.mark.parametrize("frame_num", PROBE_FRAMES)
def test_homography_places_static_scenery_within_a_pixel(registered, frame_num):
    homographies, frames, _ = registered
    source, target = static_correspondences(frames, frame_num)
    error = residual(homographies["homography"][frame_num], source, target)
    assert error < MAX_RESIDUAL_PX, f"frame {frame_num}: {error:.2f} px"


def test_translation_model_cannot_follow_this_camera(registered):
    """Documents why the default is the homography.

    This is not a failing test for the translation model -- it is correct as a translation
    estimator. It records that a translation is the wrong *model* for a camera that pans to
    follow play, which is the finding that retired the original acceptance criterion.
    """
    homographies, frames, _ = registered
    errors = {}
    for frame_num in PROBE_FRAMES:
        source, target = static_correspondences(frames, frame_num)
        errors[frame_num] = {
            name: residual(homographies[name][frame_num], source, target)
            for name in ("homography", "translation")
        }

    late = [e for n, e in errors.items() if n >= 150]
    assert all(e["translation"] > 20 * e["homography"] for e in late), errors
    assert np.median([e["translation"] for e in late]) > 20


def test_every_frame_is_registered(registered):
    _, _, models = registered
    assert models["homography"].frames_without_estimate == 0


def test_reference_frame_is_the_identity(registered):
    homographies, _, _ = registered
    assert np.allclose(homographies["homography"][0], np.eye(3))


# -- end to end --------------------------------------------------------------------------


def test_pipeline_runs_on_the_real_clip(tmp_path):
    """Full run against the committed stub: no impossible speeds, teams roughly balanced."""
    stub = Path(__file__).resolve().parent.parent / "stubs" / "track_stubs.pkl"
    if not stub.exists():
        pytest.skip("track stub missing")

    import main

    args = main.parse_args([
        "--input", str(VIDEO),
        "--output", str(tmp_path / "out.mp4"),
        "--track-stub", str(stub),
        "--use-stub",
        "--no-video",
        "--report", str(tmp_path / "report.json"),
    ])
    report = main.run(args)

    assert report["frames_analysed"] == 750
    assert report["fps"] == pytest.approx(25.0, abs=0.01)
    assert report["camera_frames_without_estimate"] == 0
    assert all(row["top_speed_kmh"] <= 36.0 for row in report["tracks"])

    shares = report["possession"]
    assert shares["team_1_possession"] + shares["team_2_possession"] == pytest.approx(1.0)
