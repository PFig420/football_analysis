"""The detection -> tracks adapter, exercised against real ultralytics objects.

No weights needed: a real `ultralytics.engine.results.Boxes` and the real BYTETracker are
enough to check the parts that actually broke -- that fields are read by name rather than
by tuple position, that goalkeepers keep their own class, and that the ball is picked by
confidence and kept out of person tracking.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

from trackers import Tracker
from trackers.tracker import BALL_TRACK_ID

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
ORIG_SHAPE = (1080, 1920)


class FakeResults:
    """Stand-in for `ultralytics.engine.results.Results`, carrying real `Boxes`."""

    def __init__(self, xyxy, conf, cls):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        data = np.column_stack(
            [xyxy, np.asarray(conf, dtype=np.float32).reshape(-1), np.asarray(cls, dtype=np.float32).reshape(-1)]
        ) if len(xyxy) else np.empty((0, 6), dtype=np.float32)
        self.boxes = Boxes(torch.as_tensor(data, dtype=torch.float32), orig_shape=ORIG_SHAPE)
        self.names = CLASS_NAMES


@pytest.fixture
def detection():
    return FakeResults(
        xyxy=[
            [100, 100, 120, 160],  # player
            [300, 100, 320, 160],  # player
            [500, 100, 520, 160],  # goalkeeper
            [700, 100, 720, 160],  # referee
            [110, 150, 118, 158],  # ball, confident
            [900, 400, 908, 408],  # ball, not confident
        ],
        conf=[0.9, 0.88, 0.85, 0.8, 0.92, 0.15],
        cls=[2, 2, 1, 3, 0, 0],
    )


def test_objects_land_in_their_own_buckets(detection):
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    tracker.append_detection(tracks, detection)

    assert len(tracks["players"][0]) == 2
    assert len(tracks["goalkeepers"][0]) == 1, "goalkeepers must not be relabelled as players"
    assert len(tracks["referees"][0]) == 1
    assert len(tracks["ball"][0]) == 1


def test_ball_is_the_most_confident_detection(detection):
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    tracker.append_detection(tracks, detection)

    ball = tracks["ball"][0][BALL_TRACK_ID]
    assert ball["confidence"] == pytest.approx(0.92)
    assert ball["bbox"][0] == pytest.approx(110)
    assert ball["interpolated"] is False


def test_track_ids_are_plain_ints(detection):
    """Track IDs become dict keys and JSON fields; numpy ints leak into both."""
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    tracker.append_detection(tracks, detection)

    for track_id in tracks["players"][0]:
        assert isinstance(track_id, int) and not isinstance(track_id, np.integer)


def test_identities_persist_across_frames():
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    for i in range(5):
        tracker.append_detection(
            tracks,
            FakeResults(
                xyxy=[[100 + i * 2, 100, 120 + i * 2, 160], [300, 100, 320, 160]],
                conf=[0.9, 0.9],
                cls=[2, 2],
            ),
        )

    first, last = set(tracks["players"][0]), set(tracks["players"][4])
    assert first and first == last, "a steadily moving player should keep its track ID"


def test_frames_with_no_detections_are_still_recorded():
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    tracker.append_detection(tracks, FakeResults(xyxy=np.empty((0, 4)), conf=[], cls=[]))

    assert len(tracks["players"]) == 1
    assert tracks["players"][0] == {}
    assert tracks["ball"][0] == {}


def test_predict_yields_the_boxes_type_this_adapter_reads():
    """Close the loop between `model.predict` and the adapter.

    The tests above feed `Boxes` in directly; this one checks that a real ultralytics model
    actually hands back that same type, so the two halves meet. Built from a YAML config
    with random weights, so it needs no download -- the predictions are meaningless and
    deliberately unused.
    """
    from ultralytics import YOLO

    from config import TrackerConfig

    model = YOLO("yolov8n.yaml")
    model.model.names = CLASS_NAMES

    tracker = Tracker(config=TrackerConfig(confidence=0.01), fps=25.0)
    tracker._model = model

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(2)]

    results = list(tracker.detect_batches([frames]))
    assert len(results) == 2
    for result in results:
        assert isinstance(result.boxes, Boxes)
        assert result.names[2] == "player"


def test_ball_is_excluded_from_person_tracking(detection):
    """Feeding the ball to ByteTrack wastes an identity slot and can steal one from a player."""
    tracker = Tracker()
    tracks = tracker.empty_tracks()
    tracker.append_detection(tracks, detection)

    people = sum(len(tracks[name][0]) for name in ("players", "goalkeepers", "referees"))
    assert people == 4
