import numpy as np
import pytest

from config import BallConfig
from player_ball_assigner import NO_PLAYER, PlayerBallAssigner, PossessionTracker
from trackers import Tracker
from trackers.tracker import BALL_TRACK_ID


def test_closest_player_in_range_gets_the_ball():
    players = {
        1: {"bbox": [100, 100, 120, 160]},
        2: {"bbox": [300, 100, 320, 160]},
    }
    assert PlayerBallAssigner().assign_ball_to_player(players, [115, 155, 125, 165]) == 1


def test_no_player_in_range_returns_sentinel():
    players = {1: {"bbox": [100, 100, 120, 160]}}
    assert PlayerBallAssigner().assign_ball_to_player(players, [900, 900, 910, 910]) == NO_PLAYER


def test_empty_inputs_are_safe():
    assigner = PlayerBallAssigner()
    assert assigner.assign_ball_to_player({}, [1, 2, 3, 4]) == NO_PLAYER
    assert assigner.assign_ball_to_player({1: {"bbox": [0, 0, 1, 1]}}, None) == NO_PLAYER


def test_threshold_scales_with_resolution():
    """A 70 px reach at 1080p is 35 px at 540p; the original was resolution-blind."""
    assert PlayerBallAssigner(BallConfig(), scale=0.5).max_distance == pytest.approx(35.0)


def test_unmeasured_first_frame_does_not_raise():
    """The original did `team_ball_control[-1]` on an empty list -- IndexError."""
    possession = PossessionTracker()
    possession.add(0, measured=False)
    assert possession.shares_until(0) == (0.0, 0.0, 0.0)


def test_shares_ignore_unmeasured_frames():
    possession = PossessionTracker()
    for _ in range(3):
        possession.add(1, measured=True)
    for _ in range(1):
        possession.add(2, measured=True)
    for _ in range(96):
        possession.add(0, measured=False)

    one, two, coverage = possession.shares_until(99)
    assert one == pytest.approx(0.75)
    assert two == pytest.approx(0.25)
    assert coverage == pytest.approx(0.04)


def test_shares_sum_to_one_when_anything_is_measured():
    possession = PossessionTracker()
    for team in (1, 2, 1, 2, 1):
        possession.add(team, measured=True)
    one, two, _ = possession.shares_until(4)
    assert one + two == pytest.approx(1.0)


def test_unmeasured_frames_carry_the_last_holder_for_display():
    possession = PossessionTracker()
    possession.add(2, measured=True)
    possession.add(0, measured=False)
    assert possession.team_per_frame == [2, 2]
    assert possession.measured == [True, False]


# -- ball interpolation ----------------------------------------------------------------


def ball_frames(detected_at: set[int], n: int, confidence: float = 0.9) -> list[dict]:
    return [
        {BALL_TRACK_ID: {"bbox": [i, 0, i + 5, 5], "confidence": confidence}}
        if i in detected_at
        else {}
        for i in range(n)
    ]


def test_short_gaps_are_interpolated_and_flagged():
    tracks = ball_frames({0, 4}, 5)
    out = Tracker().interpolate_ball_positions(tracks, max_gap_frames=12)

    assert all(frame for frame in out), "a 3-frame gap should be filled"
    assert out[0][BALL_TRACK_ID]["interpolated"] is False
    assert out[2][BALL_TRACK_ID]["interpolated"] is True


def test_long_gaps_are_left_empty():
    """The original filled every gap however long; 63% of the sample clip was guesswork."""
    tracks = ball_frames({0, 100}, 101)
    out = Tracker().interpolate_ball_positions(tracks, max_gap_frames=12)

    filled = sum(1 for frame in out if frame)
    assert filled < 30, f"a 99-frame gap was bridged ({filled} frames filled)"
    assert out[0][BALL_TRACK_ID]["interpolated"] is False


def test_low_confidence_detections_are_discarded():
    tracks = ball_frames({0, 1, 2}, 3, confidence=0.05)
    out = Tracker().interpolate_ball_positions(tracks, max_gap_frames=12)
    assert all(not frame for frame in out)


# Ball selection by confidence is covered against real ultralytics Boxes in
# tests/test_detection_adapter.py::test_ball_is_the_most_confident_detection.
