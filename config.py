"""Central configuration.

Every clip-specific constant lives here rather than in a module body, so the pipeline can
run on a clip that is not 1920x1080 at 24 fps without a code edit. Overlay geometry is
expressed as a fraction of the frame so it scales with resolution; distances that are
genuinely in pixels are scaled at use time by `Config.scale` (see `PipelineConfig.for_frame`).

Load a YAML override with `PipelineConfig.load(path)`; anything absent falls back to the
defaults below, which are the values the original tutorial used for `08fd33_4.mp4`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any

import yaml

# The resolution the original hardcoded constants were tuned for. Used only to convert those
# constants into resolution-independent fractions; nothing should compare against it.
REFERENCE_HEIGHT = 1080


@dataclass
class PitchConfig:
    """The quadrilateral used for the perspective transform.

    `pixel_vertices` are in *frame-0 pixel space* and must be given in the order
    (near-left, far-left, far-right, near-right) so that they correspond, one for one, to
    the corners of a `length_m` x `width_m` rectangle on the pitch.

    WARNING: the defaults were eyeballed from the sample clip by the original author. There
    is no record of which pitch lines they correspond to and they are almost certainly wrong
    for any other clip. `calibrated` is False to make that explicit; the pipeline logs a
    warning and the resulting speeds and distances should be treated as uncalibrated.
    """

    pixel_vertices: list[list[float]] = field(
        default_factory=lambda: [[110, 1035], [265, 275], [910, 260], [1640, 915]]
    )
    length_m: float = 23.32
    width_m: float = 68.0
    calibrated: bool = False


@dataclass
class TrackerConfig:
    """Detection, plus the ByteTrack association parameters.

    Tracking uses ultralytics' own BYTETracker. `supervision` used to provide it but removed
    `ByteTrack` in 0.31.0; ultralytics ships the same algorithm and is already a dependency.
    """

    model_path: str = "models/best.pt"
    batch_size: int = 20
    confidence: float = 0.1
    #: Ball detections below this are discarded outright. The detector is run at a low
    #: threshold to catch the ball at all; this filters the resulting false positives.
    ball_confidence: float = 0.3

    # -- ByteTrack association (ultralytics bytetrack.yaml defaults unless noted) --
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    fuse_score: bool = True
    #: How long a track survives unmatched, in **seconds** rather than frames, so the same
    #: setting means the same thing at 24 fps and 50 fps. ByteTrack's own default is 30
    #: frames, which is 1.25 s at 24 fps.
    track_buffer_seconds: float = 1.25


@dataclass
class BallConfig:
    #: Maximum gap, in seconds, that may be bridged by interpolation. Longer gaps are left
    #: empty rather than guessed. The sample clip has the ball in only 37% of frames, so
    #: without a cap most "ball positions" are straight lines between distant sightings.
    max_interpolation_seconds: float = 0.5
    #: A player must be at least this close (in pixels at the reference resolution) to be
    #: given possession.
    max_possession_distance_px: float = 70.0


@dataclass
class CameraConfig:
    """How camera motion is modelled.

    `model` is "homography" or "translation".

    **Prefer "homography".** A broadcast camera pans, tilts and zooms, and the mapping that
    induces on the image is projective. Registering frame 0 of the sample clip against frame
    200, a homography leaves a 0.62 px residual where the best possible translation leaves
    65.70 px -- an error far larger than any player movement being measured.

    "translation" is optical-flow based, costs almost nothing, and is adequate only when the
    camera is near-static (a fixed tactical camera, say). It is the original behaviour.
    """

    model: str = "homography"

    # -- homography model (SIFT + RANSAC) --
    #: Features are detected at this fraction of full resolution. Halving the frame is about
    #: four times faster and costs nothing measurable in accuracy at 1080p.
    detector_scale: float = 0.5
    max_features: int = 2000
    #: Lowe's ratio test threshold for accepting a descriptor match.
    ratio_test: float = 0.75
    min_matches: int = 25
    min_inliers: int = 20

    # -- translation model (sparse optical flow) --
    #: Corner-detection parameters for the features tracked between frames.
    max_corners: int = 500
    quality_level: float = 0.1
    min_distance: int = 7
    block_size: int = 7
    #: Re-detect features when fewer than this fraction of them survive the optical flow.
    refresh_below_fraction: float = 0.5
    #: ...or unconditionally after this many frames, so features cannot drift indefinitely.
    refresh_every_frames: int = 30
    #: RANSAC reprojection threshold, in pixels, for the global-motion fit.
    ransac_threshold: float = 3.0
    #: Per-frame translations with magnitude below this are treated as zero, to stop
    #: sub-pixel estimation noise accumulating into a phantom drift over a long clip.
    deadzone_px: float = 0.25


@dataclass
class SpeedConfig:
    #: Speed is measured over a sliding window of this many seconds, not a fixed frame count,
    #: so the measurement means the same thing at 24 fps and at 50 fps.
    window_seconds: float = 0.25
    #: Savitzky-Golay smoothing of the transformed trajectory before differencing, in
    #: seconds. Raw per-window chords accumulate jitter into the distance total.
    smoothing_seconds: float = 0.5
    #: Displacements below this (metres) within a window are treated as noise, not movement.
    noise_floor_m: float = 0.05
    #: Sanity ceiling. Samples above this are dropped and counted; the human record is
    #: about 44 km/h over a few metres, and nothing in a team sport should approach it.
    max_plausible_kmh: float = 36.0


@dataclass
class TeamConfig:
    #: Number of frames sampled across the clip to fit the two team-colour centroids.
    fit_frames: int = 30
    #: A track's team is decided by majority vote over this many observations rather than
    #: being locked in on first sight.
    vote_observations: int = 10


@dataclass
class OverlayConfig:
    """Overlay geometry as (x1, y1, x2, y2) fractions of frame width/height."""

    possession_box: tuple[float, float, float, float] = (0.703, 0.787, 0.990, 0.898)
    camera_box: tuple[float, float, float, float] = (0.0, 0.0, 0.260, 0.093)
    draw_camera_movement: bool = True
    draw_speed_distance: bool = True
    draw_possession: bool = True

    def __post_init__(self) -> None:
        # YAML round-trips tuples as lists; normalise so equality and unpacking behave.
        self.possession_box = tuple(self.possession_box)
        self.camera_box = tuple(self.camera_box)


@dataclass
class PipelineConfig:
    pitch: PitchConfig = field(default_factory=PitchConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    #: Set by `for_frame`; multiplies pixel-denominated constants and font sizes so they
    #: mean the same thing at 720p as at 1080p.
    scale: float = 1.0

    def for_frame(self, width: int, height: int) -> "PipelineConfig":
        """Return a copy whose `scale` matches this clip's resolution."""
        return replace(self, scale=height / REFERENCE_HEIGHT)

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        sections = {
            "pitch": PitchConfig,
            "tracker": TrackerConfig,
            "ball": BallConfig,
            "camera": CameraConfig,
            "speed": SpeedConfig,
            "team": TeamConfig,
            "overlay": OverlayConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            if name in data:
                kwargs[name] = klass(**data[name])
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        return cls(**kwargs)

    def dump(self, path: str | Path) -> None:
        # `scale` is derived from the clip at run time, not a setting, so it is not written.
        data = {k: v for k, v in asdict(self).items() if k != "scale"}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
