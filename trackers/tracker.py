"""Detection and tracking.

Changes from the original:

* **Goalkeepers keep their own class.** They used to be relabelled as outfield players, which
  then fed them into kit-colour clustering they could never match -- the root cause of the
  `if player_id == 91` fudge that used to live in `team_assigner`.
* **Tracking comes from ultralytics, not supervision.** `supervision` removed `ByteTrack`
  in 0.31.0, so the original code raises AttributeError before processing a frame on a
  current install. ultralytics ships the same algorithm and was already a dependency, so
  `supervision` has been dropped entirely.
* **Detections are read by named field** (`.xyxy`, `.conf`, `.cls`) instead of by tuple
  position (`frame_detection[0]`, `[3]`, `[4]`), which was coupled to a library version.
* **The ball is the highest-confidence detection in the frame**, not whichever came last, and
  its confidence is retained. Interpolation is capped, and interpolated frames are flagged
  so possession can exclude them.
* **Stubs carry a header** describing the video and model they came from, and are refused if
  it does not match. The original rewrote the stub whenever a path was given and reloaded it
  by default, so changes to detection silently did nothing.
* The `ultralytics` import is deferred, so everything downstream of detection can be tested
  without torch installed.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import pandas as pd

from config import TrackerConfig
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

logger = logging.getLogger(__name__)

STUB_VERSION = 2
BALL_TRACK_ID = 1

#: Tracked object classes, in the order they are drawn.
OBJECT_CLASSES = ("players", "goalkeepers", "referees", "ball")

#: Model class name -> our track bucket.
CLASS_TO_BUCKET = {
    "player": "players",
    "goalkeeper": "goalkeepers",
    "referee": "referees",
    "ball": "ball",
}


@dataclass(frozen=True)
class StubHeader:
    """Identifies the inputs a stub was produced from."""

    version: int
    video_name: str
    video_size: int
    frame_count: int
    model_name: str

    @classmethod
    def build(cls, video_path, model_path, frame_count: int) -> "StubHeader":
        video = Path(video_path)
        return cls(
            version=STUB_VERSION,
            video_name=video.name,
            video_size=video.stat().st_size if video.exists() else -1,
            frame_count=frame_count,
            model_name=Path(model_path).name,
        )

    def mismatch(self, other: "StubHeader") -> str | None:
        for field_name in ("version", "video_name", "video_size", "model_name"):
            mine, theirs = getattr(self, field_name), getattr(other, field_name)
            if mine != theirs:
                return f"{field_name}: stub has {theirs!r}, this run has {mine!r}"
        return None


def save_stub(path, tracks: dict, header: StubHeader) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"header": asdict(header), "tracks": tracks}, f)
    logger.info("wrote stub %s (%d frames)", path, header.frame_count)


def load_stub(path, header: StubHeader | None = None) -> dict | None:
    """Load a stub, or return None if it is absent or does not match this run.

    NOTE: this unpickles a file from disk. Only load stubs you generated yourself.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "tracks" not in payload:
        # The tutorial's original format: a bare {"players": [...], ...} dict with no
        # provenance. Accept it so the committed fixture keeps working, but say so.
        logger.warning(
            "%s is a legacy headerless stub; cannot verify it matches this video or model", path
        )
        return payload

    if header is not None:
        stored = StubHeader(**payload["header"])
        problem = header.mismatch(stored)
        if problem:
            raise ValueError(
                f"stub {path} does not match this run ({problem}). "
                f"Delete it or pass --no-stub to regenerate."
            )
    return payload["tracks"]


class Tracker:
    def __init__(
        self,
        model_path: str | None = None,
        config: TrackerConfig | None = None,
        fps: float = 30.0,
    ):
        self.config = config or TrackerConfig()
        self.model_path = model_path or self.config.model_path
        #: Only used to express the track buffer in seconds rather than frames.
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._model = None
        self._tracker = None

    # -- lazy heavyweight imports ------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"model weights not found at {self.model_path}. See the README for the "
                    f"download link, or pass --model."
                )
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    @property
    def byte_tracker(self):
        """ByteTrack, from ultralytics.

        This used to come from `supervision`, which removed `ByteTrack` in 0.31.0 -- the
        original code raises AttributeError before processing a frame on any current
        install. ultralytics ships the same algorithm, is already a dependency for
        detection, and is maintained, so `supervision` is no longer needed at all.
        """
        if self._tracker is None:
            from ultralytics.trackers import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace

            self._tracker = BYTETracker(
                IterableSimpleNamespace(
                    tracker_type="bytetrack",
                    track_high_thresh=self.config.track_high_thresh,
                    track_low_thresh=self.config.track_low_thresh,
                    new_track_thresh=self.config.new_track_thresh,
                    match_thresh=self.config.match_thresh,
                    fuse_score=self.config.fuse_score,
                    # ByteTrack counts this in frames; we specify it in seconds so the
                    # behaviour does not change with the clip's frame rate.
                    track_buffer=max(1, int(round(self.config.track_buffer_seconds * self.fps))),
                )
            )
        return self._tracker

    def reset(self) -> None:
        """Clear tracking state so one Tracker can process several clips."""
        self._tracker = None

    # -- detection ---------------------------------------------------------------------

    def detect_batches(self, batches: Iterable[list[np.ndarray]]) -> Iterator:
        for batch in batches:
            yield from self.model.predict(batch, conf=self.config.confidence, verbose=False)

    @staticmethod
    def empty_tracks() -> dict:
        return {name: [] for name in OBJECT_CLASSES}

    def tracks_from_detections(self, detections: Iterable) -> dict:
        """Turn a stream of per-frame detections into per-frame track dicts."""
        tracks = self.empty_tracks()
        for detection in detections:
            self.append_detection(tracks, detection)
        return tracks

    def append_detection(self, tracks: dict, detection, frame: np.ndarray | None = None) -> None:
        """Append one frame's detections to `tracks`, in place.

        Separate from `tracks_from_detections` so the pipeline can interleave detection with
        its other per-frame work in a single streaming pass over the video.

        Reads ultralytics' `Boxes` directly. The previous version went through
        `supervision.Detections` and then indexed the result by tuple position
        (`frame_detection[0]`, `[3]`, `[4]`), which was coupled to the installed
        supervision version for no benefit.
        """
        class_names = detection.names
        boxes = detection.boxes
        boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes

        for name in OBJECT_CLASSES:
            tracks[name].append({})
        frame_num = len(tracks["players"]) - 1

        # The ball is not tracked: ByteTrack is tuned for people, and a single ball needs no
        # identity. Take the most confident detection in the frame rather than the last one.
        ball = self._best_ball(boxes, class_names)
        if ball is not None:
            bbox, confidence = ball
            tracks["ball"][frame_num][BALL_TRACK_ID] = {
                "bbox": bbox,
                "confidence": confidence,
                "interpolated": False,
            }

        if len(boxes) == 0:
            return

        # Keep the ball out of the association step, where it would occupy a person identity.
        is_person = np.array([class_names[int(c)] != "ball" for c in boxes.cls], dtype=bool)
        if not is_person.any():
            return

        tracked = self.byte_tracker.update(boxes[is_person], frame)
        for row in tracked:
            # Rows are [x1, y1, x2, y2, track_id, score, cls, idx].
            x1, y1, x2, y2, track_id, _score, class_id = row[:7]
            bucket = CLASS_TO_BUCKET.get(class_names[int(class_id)])
            if bucket is None or bucket == "ball":
                continue
            tracks[bucket][frame_num][int(track_id)] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            }

    @staticmethod
    def _best_ball(boxes, class_names):
        """Highest-confidence ball in the frame, or None."""
        best = None
        for bbox, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if class_names[int(class_id)] != "ball":
                continue
            if best is None or confidence > best[1]:
                best = ([float(v) for v in bbox], float(confidence))
        return best

    def get_object_tracks(
        self,
        batches: Iterable[list[np.ndarray]],
        stub_path: str | None = None,
        read_from_stub: bool = False,
        header: StubHeader | None = None,
    ) -> dict:
        if read_from_stub:
            cached = load_stub(stub_path, header)
            if cached is not None:
                logger.info("loaded tracks from stub %s", stub_path)
                return cached

        tracks = self.tracks_from_detections(self.detect_batches(batches))

        if stub_path:
            save_stub(
                stub_path,
                tracks,
                header or StubHeader(STUB_VERSION, "unknown", -1, len(tracks["players"]), "unknown"),
            )
        return tracks

    # -- derived quantities ------------------------------------------------------------

    @staticmethod
    def add_position_to_tracks(tracks: dict) -> None:
        """Ball position is its centre; a person's position is their feet."""
        for obj, object_tracks in tracks.items():
            for frame_tracks in object_tracks:
                for info in frame_tracks.values():
                    bbox = info.get("bbox")
                    if bbox is None or not np.isfinite(bbox).all():
                        continue
                    info["position"] = (
                        get_center_of_bbox(bbox) if obj == "ball" else get_foot_position(bbox)
                    )

    def interpolate_ball_positions(
        self, ball_tracks: list[dict], max_gap_frames: int
    ) -> list[dict]:
        """Fill gaps in the ball track up to `max_gap_frames`; leave longer ones empty.

        The original interpolated every gap, however long, and marked nothing. On the sample
        clip the ball is detected in 37% of frames, so most "positions" were straight lines
        between sightings that could be seconds apart.
        """
        max_gap = max(1, int(max_gap_frames))

        rows, detected = [], []
        for frame_tracks in ball_tracks:
            entry = frame_tracks.get(BALL_TRACK_ID)
            confident = (
                entry is not None and entry.get("confidence", 1.0) >= self.config.ball_confidence
            )
            rows.append(entry["bbox"] if confident else [np.nan] * 4)
            detected.append(confident)

        frame = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        filled = frame.interpolate(limit=max_gap, limit_area="inside").bfill(limit=max_gap)

        if filled.isna().all().all():
            logger.warning("the ball was never detected above the confidence threshold")

        out: list[dict] = []
        for i, (values, was_detected) in enumerate(zip(filled.to_numpy().tolist(), detected)):
            if not np.isfinite(values).all():
                out.append({})
                continue
            out.append(
                {
                    BALL_TRACK_ID: {
                        "bbox": values,
                        "interpolated": not was_detected,
                        "confidence": ball_tracks[i].get(BALL_TRACK_ID, {}).get("confidence", 0.0),
                    }
                }
            )

        kept = sum(1 for f in out if f)
        logger.info(
            "ball: detected in %d/%d frames (%.0f%%), usable after capped interpolation %d (%.0f%%)",
            sum(detected), len(detected), 100 * np.mean(detected) if detected else 0,
            kept, 100 * kept / len(out) if out else 0,
        )
        return out

    # -- drawing -----------------------------------------------------------------------

    @staticmethod
    def draw_ellipse(frame, bbox, color, track_id=None, scale: float = 1.0):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        color = tuple(int(c) for c in color)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=max(1, int(round(2 * scale))),
            lineType=cv2.LINE_4,
        )

        if track_id is None:
            return frame

        box_w, box_h = int(40 * scale), int(20 * scale)
        x1 = x_center - box_w // 2
        y1 = int(y2 - box_h // 2 + 15 * scale)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, cv2.FILLED)

        text = str(track_id)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
        cv2.putText(
            frame,
            text,
            (x1 + (box_w - text_w) // 2, y1 + (box_h + text_h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale,
            (0, 0, 0),
            max(1, int(round(2 * scale))),
        )
        return frame

    @staticmethod
    def draw_triangle(frame, bbox, color, scale: float = 1.0):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        size = max(6, int(10 * scale))
        points = np.array([[x, y], [x - size, y - 2 * size], [x + size, y - 2 * size]])
        cv2.drawContours(frame, [points], 0, tuple(int(c) for c in color), cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), max(1, int(round(2 * scale))))
        return frame

    #: Original spelling, retained for any external caller.
    draw_traingle = draw_triangle

    def draw_frame(self, frame, tracks: dict, frame_num: int, scale: float = 1.0):
        """Annotate a single frame in place. Per-frame so the pipeline can stream."""
        for track_id, player in tracks["players"][frame_num].items():
            color = player.get("team_color", (0, 0, 255))
            self.draw_ellipse(frame, player["bbox"], color, track_id, scale)
            if player.get("has_ball"):
                self.draw_triangle(frame, player["bbox"], (0, 0, 255), scale)

        for track_id, keeper in tracks.get("goalkeepers", [{}] * (frame_num + 1))[frame_num].items():
            color = keeper.get("team_color", (255, 255, 255))
            self.draw_ellipse(frame, keeper["bbox"], color, track_id, scale)

        for referee in tracks["referees"][frame_num].values():
            self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), None, scale)

        for ball in tracks["ball"][frame_num].values():
            # Hollow marker when the position was interpolated rather than detected.
            color = (0, 200, 255) if ball.get("interpolated") else (0, 255, 0)
            self.draw_triangle(frame, ball["bbox"], color, scale)

        return frame
