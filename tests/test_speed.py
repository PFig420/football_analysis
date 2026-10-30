import numpy as np
import pytest

from config import SpeedConfig
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def straight_line_track(n_frames: int, metres_per_frame: float) -> dict:
    """One player walking in a straight line at a constant rate."""
    return {
        "players": [
            {1: {"bbox": [0, 0, 10, 10], "position_transformed": [i * metres_per_frame, 0.0]}}
            for i in range(n_frames)
        ]
    }


def test_speed_is_correct_at_24_fps():
    # 0.25 m/frame at 24 fps = 6 m/s = 21.6 km/h
    tracks = straight_line_track(120, 0.25)
    SpeedAndDistanceEstimator(24.0).add_speed_and_distance_to_tracks(tracks)
    speeds = [i["speed"] for f in tracks["players"] for i in f.values() if "speed" in i]
    assert np.mean(speeds) == pytest.approx(21.6, rel=0.02)


def test_frame_rate_comes_from_the_clip_not_a_constant():
    """Identical per-frame displacements mean faster motion at a higher frame rate.

    The original hardcoded 24 fps, so on a 50 fps clip every speed came out at 24/50 of the
    truth -- and every distance was unaffected, which made the error easy to miss.
    """
    # 0.1 m/frame is 8.6 km/h at 24 fps and 18 km/h at 50 fps -- both under the cap, so the
    # ratio reflects the frame rate rather than the plausibility filter.
    tracks_24 = straight_line_track(240, 0.1)
    tracks_50 = straight_line_track(240, 0.1)

    SpeedAndDistanceEstimator(24.0).add_speed_and_distance_to_tracks(tracks_24)
    SpeedAndDistanceEstimator(50.0).add_speed_and_distance_to_tracks(tracks_50)

    def mean_speed(tracks):
        return np.mean([i["speed"] for f in tracks["players"] for i in f.values() if "speed" in i])

    assert mean_speed(tracks_50) / mean_speed(tracks_24) == pytest.approx(50 / 24, rel=0.03)


def test_distance_matches_the_straight_line_length():
    tracks = straight_line_track(241, 0.1)  # 240 steps * 0.1 m = 24 m
    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks(tracks)
    summary = estimator.summaries[("players", 1)]
    assert summary.distance_m == pytest.approx(24.0, rel=0.05)


def test_jitter_does_not_inflate_distance():
    """A stationary player with measurement noise must not accumulate distance.

    Summing raw per-window chords, as the original did, turns jitter into kilometres.
    """
    rng = np.random.default_rng(1)
    tracks = {
        "players": [
            {1: {"bbox": [0, 0, 10, 10], "position_transformed": list(rng.normal(0, 0.05, 2))}}
            for _ in range(240)
        ]
    }
    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks(tracks)
    assert estimator.summaries[("players", 1)].distance_m < 2.0


def test_implausible_speeds_are_dropped_and_counted():
    # 5 m/frame at 24 fps = 432 km/h
    tracks = straight_line_track(120, 5.0)
    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks(tracks)

    assert estimator.dropped_total > 0
    speeds = [i["speed"] for f in tracks["players"] for i in f.values() if "speed" in i]
    assert all(s <= SpeedConfig().max_plausible_kmh for s in speeds)


def test_gaps_break_the_trajectory_rather_than_teleporting():
    """A track that leaves the quad and returns elsewhere must not bank the jump."""
    frames = []
    for i in range(60):
        frames.append({1: {"bbox": [0, 0, 10, 10], "position_transformed": [i * 0.05, 0.0]}})
    for _ in range(30):
        frames.append({1: {"bbox": [0, 0, 10, 10], "position_transformed": None}})
    for i in range(60):
        frames.append({1: {"bbox": [0, 0, 10, 10], "position_transformed": [50.0 + i * 0.05, 0.0]}})

    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks({"players": frames})
    summary = estimator.summaries[("players", 1)]

    assert summary.distance_m < 10.0, "the 47 m gap was counted as running"
    assert summary.coverage == pytest.approx(120 / 150)


def test_summary_reports_span_and_coverage():
    tracks = straight_line_track(48, 0.1)
    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks(tracks)
    row = estimator.summary_rows()[0]

    assert row["track_id"] == 1
    assert row["span_seconds"] == pytest.approx(2.0, abs=0.05)
    assert row["coverage"] == 1.0


def test_rejects_invalid_fps():
    with pytest.raises(ValueError):
        SpeedAndDistanceEstimator(0)


def test_real_stub_produces_no_impossible_speeds(track_stub):
    """Acceptance criterion: nothing above 36 km/h on the sample clip."""
    from camera_movement_estimator import CameraModel
    from trackers import Tracker
    from view_transformer import ViewTransformer

    tracks = {"players": track_stub["players"]}
    Tracker.add_position_to_tracks(tracks)
    # Identity homographies: this test is about the speed chain, not the camera.
    CameraModel.add_adjust_positions_to_tracks(
        tracks, np.tile(np.eye(3), (len(tracks["players"]), 1, 1))
    )
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    estimator = SpeedAndDistanceEstimator(24.0)
    estimator.add_speed_and_distance_to_tracks(tracks)

    speeds = [i["speed"] for f in tracks["players"] for i in f.values() if "speed" in i]
    assert speeds, "no speeds were produced from the fixture"
    assert max(speeds) <= SpeedConfig().max_plausible_kmh

    # Nobody covers more than 14 km per 90 minutes; the clip is 31 s.
    ceiling_m = 14_000 * (750 / 24) / 5400
    assert max(s.distance_m for s in estimator.summaries.values()) < ceiling_m
