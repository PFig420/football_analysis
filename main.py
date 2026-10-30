"""Football video analysis pipeline.

Two bounded-memory passes over the clip:

  pass 1  decode -> detect + track -> camera drift -> harvest kit crops
  (no frames held)  ball interpolation, positions, perspective, teams, speed, possession
  pass 2  decode -> draw -> encode

The original decoded the whole clip into a list and kept an annotated copy alongside it:
about 9 GB for 31 seconds of 1080p. Nothing here grows with clip length except the track
dictionaries.

    python main.py --input input_videos/08fd33_4.mp4 --output output_videos/out.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from camera_movement_estimator import CameraModel, make_camera_model
from config import PipelineConfig
from player_ball_assigner import NO_PLAYER, PlayerBallAssigner, PossessionTracker
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TEAM_A, TEAM_B, UNKNOWN_TEAM, TeamAssigner
from trackers import StubHeader, Tracker, load_stub
from utils import iter_batches, iter_frames, probe_video, video_writer
from view_transformer import ViewTransformer

logger = logging.getLogger("pipeline")


# -- pass 1 ----------------------------------------------------------------------------


def analyse(args, cfg: PipelineConfig, meta) -> tuple[dict, np.ndarray, dict, int]:
    """Decode once: detect, track, measure camera drift, collect kit crops."""
    tracker = Tracker(args.model, cfg.tracker, fps=meta.fps)
    camera = make_camera_model(cfg.camera)
    assigner = TeamAssigner(cfg.team)

    header = StubHeader.build(args.input, args.model, meta.frame_count)
    cached = load_stub(args.track_stub, header) if args.use_stub else None
    detecting = cached is None
    if detecting and args.use_stub:
        logger.info("no usable track stub; running detection")

    tracks = cached if cached is not None else tracker.empty_tracks()
    tracks.setdefault("goalkeepers", [{} for _ in tracks["players"]])

    # One 3x3 homography per frame, mapping that frame into the reference view.
    camera_movement: list = []
    crops: dict[int, list] = defaultdict(list)
    frame_num = 0
    exhausted = False

    for batch in iter_batches(args.input, cfg.tracker.batch_size, limit=args.max_frames):
        if detecting:
            for frame, detection in zip(batch, tracker.detect_batches([batch])):
                tracker.append_detection(tracks, detection, frame)

        for frame in batch:
            if frame_num >= len(tracks["players"]):
                logger.warning(
                    "stub covers %d frames but the video has more; analysing the first %d",
                    len(tracks["players"]),
                    len(tracks["players"]),
                )
                exhausted = True
                break
            camera_movement.append(camera.update(frame))

            # Keep a few small crops per track rather than whole frames: this is what lets
            # team colours be fitted after the pass instead of during a second one.
            for track_id, info in tracks["players"][frame_num].items():
                if len(crops[track_id]) < cfg.team.vote_observations:
                    crop = assigner.crop_player(frame, info["bbox"])
                    if crop is not None:
                        crops[track_id].append(crop)
            frame_num += 1

        if exhausted:
            break

    logger.info(
        "pass 1: %d frames, camera pan %s px at frame centre, %d frame(s) without an estimate",
        frame_num,
        np.round(
            CameraModel.pan_at(camera_movement[-1], meta.width, meta.height)
            if camera_movement
            else (0.0, 0.0),
            1,
        ).tolist(),
        camera.frames_without_estimate,
    )
    return tracks, np.array(camera_movement, dtype=float), crops, frame_num, camera


# -- derived quantities ------------------------------------------------------------------


def truncate(tracks: dict, n: int) -> None:
    for name in tracks:
        del tracks[name][n:]


def assign_teams(tracks: dict, crops: dict, assigner: TeamAssigner) -> None:
    colors_per_track = {}
    for track_id, track_crops in crops.items():
        colors = [c for c in (assigner.color_from_crop(crop) for crop in track_crops) if c is not None]
        if colors:
            colors_per_track[track_id] = colors

    if not colors_per_track:
        logger.warning("no usable player crops; teams cannot be assigned")
        return

    assigner.fit_colors([c for colors in colors_per_track.values() for c in colors])
    decided = assigner.vote_from_colors(colors_per_track)

    for frame_tracks in tracks["players"]:
        for track_id, info in frame_tracks.items():
            team = decided.get(track_id, UNKNOWN_TEAM)
            info["team"] = team
            info["team_color"] = assigner.color_of(team) if team else (200, 200, 200)

    counts = {t: sum(1 for v in decided.values() if v == t) for t in (TEAM_A, TEAM_B)}
    logger.info("teams: %d tracks on team 1, %d on team 2", counts[TEAM_A], counts[TEAM_B])


def compute_possession(tracks: dict, cfg: PipelineConfig, n_frames: int) -> PossessionTracker:
    assigner = PlayerBallAssigner(cfg.ball, cfg.scale)
    possession = PossessionTracker()

    for frame_num in range(n_frames):
        ball = tracks["ball"][frame_num].get(1)
        holder = NO_PLAYER
        if ball is not None:
            holder = assigner.assign_ball_to_player(tracks["players"][frame_num], ball["bbox"])

        team = UNKNOWN_TEAM
        if holder != NO_PLAYER:
            info = tracks["players"][frame_num][holder]
            info["has_ball"] = True
            team = info.get("team", UNKNOWN_TEAM)

        # Only a detected ball with an identified holder is evidence. Interpolated frames
        # are displayed but never counted -- see player_ball_assigner for why.
        measured = (
            ball is not None
            and not ball.get("interpolated", False)
            and holder != NO_PLAYER
            and team != UNKNOWN_TEAM
        )
        possession.add(team, measured)

    return possession


# -- pass 2 ----------------------------------------------------------------------------


def render(args, cfg: PipelineConfig, meta, tracks, camera_movement, possession, speed) -> None:
    tracker = Tracker(args.model, cfg.tracker, fps=meta.fps)
    camera = CameraModel()  # drawing only; needs no detector
    n_frames = len(tracks["players"])

    with video_writer(args.output, meta) as writer:
        for frame_num, frame in enumerate(iter_frames(args.input, limit=n_frames)):
            tracker.draw_frame(frame, tracks, frame_num, cfg.scale)

            if cfg.overlay.draw_speed_distance:
                speed.draw(frame, tracks, frame_num, cfg.scale)
            if cfg.overlay.draw_camera_movement and frame_num < len(camera_movement):
                camera.draw(frame, camera_movement[frame_num], cfg.scale, cfg.overlay.camera_box)
            if cfg.overlay.draw_possession:
                possession.draw(frame, frame_num, cfg.overlay.possession_box, cfg.scale)

            writer.write(frame)

    logger.info("wrote %s", args.output)


# -- entry point -------------------------------------------------------------------------


def run(args) -> dict:
    cfg = PipelineConfig.load(args.config)
    meta = probe_video(args.input)
    cfg = cfg.for_frame(meta.width, meta.height)
    logger.info("input %s", meta)

    tracks, camera_movement, crops, n_frames, camera = analyse(args, cfg, meta)
    if n_frames == 0:
        raise RuntimeError("no frames decoded")
    truncate(tracks, n_frames)

    tracker = Tracker(args.model, cfg.tracker, fps=meta.fps)

    # Interpolate before deriving positions: the original replaced the ball dicts afterwards,
    # which silently discarded the position fields it had just computed for them.
    max_gap = max(1, int(round(cfg.ball.max_interpolation_seconds * meta.fps)))
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"], max_gap)

    tracker.add_position_to_tracks(tracks)
    CameraModel.add_adjust_positions_to_tracks(tracks, camera_movement)

    view = ViewTransformer(cfg.pitch)
    view.add_transformed_position_to_tracks(tracks)
    logger.info(view.coverage_report())

    assigner = TeamAssigner(cfg.team)
    assign_teams(tracks, crops, assigner)
    if assigner.kmeans is not None:
        assigner.assign_goalkeeper_teams(tracks)

    speed = SpeedAndDistanceEstimator(meta.fps, cfg.speed)
    speed.add_speed_and_distance_to_tracks(tracks)
    logger.info(speed.report())

    possession = compute_possession(tracks, cfg, n_frames)
    logger.info("possession: %s", possession.summary())

    report = {
        "video": str(meta),
        "fps": meta.fps,
        "frames_analysed": n_frames,
        "pitch_calibrated": cfg.pitch.calibrated,
        "pitch_coverage": round(view.coverage, 4),
        "camera_model": cfg.camera.model,
        "camera_pan_px": np.round(
            CameraModel.pan_at(camera_movement[-1], meta.width, meta.height), 1
        ).tolist()
        if len(camera_movement)
        else [0.0, 0.0],
        "camera_frames_without_estimate": camera.frames_without_estimate,
        "possession": possession.summary(),
        "speed_summary": speed.report(),
        "tracks": speed.summary_rows(),
    }

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.report)

    if not args.no_video:
        render(args, cfg, meta, tracks, camera_movement, possession, speed)

    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input video")
    parser.add_argument("--output", default="output_videos/output_video.mp4", help="annotated video")
    parser.add_argument("--model", default="models/best.pt", help="YOLO weights")
    parser.add_argument("--config", default=None, help="YAML config overriding config.py defaults")
    parser.add_argument("--track-stub", default=None, help="cache detections here")
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="reuse --track-stub if it matches this video and model (off by default: the "
        "original defaulted to on, so edits to detection silently did nothing)",
    )
    parser.add_argument("--report", default=None, help="write a JSON report here")
    parser.add_argument("--no-video", action="store_true", help="measure only, skip rendering")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
