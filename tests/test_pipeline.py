"""End-to-end run of the real pipeline on a synthetic clip.

Uses a stub for the detection step, so this exercises every other stage -- camera drift,
interpolation, perspective, teams, speed, possession, rendering -- with no weights, no
sample video and no torch.
"""
from __future__ import annotations

import numpy as np
import pytest

import main
from config import PipelineConfig
from trackers import StubHeader, save_stub
from utils import probe_video, save_video

WIDTH, HEIGHT, N_FRAMES, FPS = 320, 240, 48, 25.0


@pytest.fixture
def clip(tmp_path):
    """A moving textured scene, so the camera estimator has something to track."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (HEIGHT + 40, WIDTH + 40, 3), dtype=np.uint8)
    frames = [base[i // 8 : i // 8 + HEIGHT, i // 8 : i // 8 + WIDTH].copy() for i in range(N_FRAMES)]
    path = tmp_path / "clip.mp4"
    save_video(frames, path, fps=FPS)
    return path


@pytest.fixture
def stub(tmp_path, clip):
    """Two players converging on a ball, plus a referee and a keeper."""
    players, referees, goalkeepers, ball = [], [], [], []
    for i in range(N_FRAMES):
        players.append(
            {
                1: {"bbox": [60 + i, 150, 76 + i, 190]},
                2: {"bbox": [200 - i, 150, 216 - i, 190]},
            }
        )
        referees.append({9: {"bbox": [150, 120, 166, 160]}})
        goalkeepers.append({5: {"bbox": [30, 150, 46, 190]}})
        # Detected only every third frame, to exercise capped interpolation.
        ball.append(
            {1: {"bbox": [70 + i, 185, 76 + i, 191], "confidence": 0.8, "interpolated": False}}
            if i % 3 == 0
            else {}
        )

    tracks = {"players": players, "goalkeepers": goalkeepers, "referees": referees, "ball": ball}
    path = tmp_path / "tracks.pkl"
    save_stub(path, tracks, StubHeader.build(clip, "models/best.pt", N_FRAMES))
    return path


@pytest.fixture
def config(tmp_path):
    """A pitch quad that actually fits this small frame."""
    cfg = PipelineConfig()
    cfg.pitch.pixel_vertices = [[10, 230], [40, 60], [280, 60], [310, 230]]
    cfg.pitch.length_m = 40.0
    cfg.pitch.width_m = 30.0
    cfg.pitch.calibrated = True
    path = tmp_path / "config.yaml"
    cfg.dump(path)
    return path


def run_pipeline(tmp_path, clip, stub, config, **overrides):
    args = main.parse_args(
        [
            "--input", str(clip),
            "--output", str(tmp_path / "out.mp4"),
            "--config", str(config),
            "--track-stub", str(stub),
            "--use-stub",
            "--report", str(tmp_path / "report.json"),
            *overrides.pop("extra", []),
        ]
    )
    return main.run(args)


def test_full_run_produces_video_and_report(tmp_path, clip, stub, config):
    report = run_pipeline(tmp_path, clip, stub, config)

    assert (tmp_path / "out.mp4").exists()
    assert (tmp_path / "report.json").exists()
    assert report["frames_analysed"] == N_FRAMES
    assert report["fps"] == pytest.approx(FPS, rel=0.01)


def test_output_matches_input_resolution_and_rate(tmp_path, clip, stub, config):
    run_pipeline(tmp_path, clip, stub, config)

    source, result = probe_video(clip), probe_video(tmp_path / "out.mp4")
    assert (result.width, result.height) == (source.width, source.height)
    assert result.fps == pytest.approx(source.fps, rel=0.01)
    assert result.frame_count == N_FRAMES


def test_possession_is_reported_with_its_coverage(tmp_path, clip, stub, config):
    report = run_pipeline(tmp_path, clip, stub, config)
    possession = report["possession"]

    shares = possession["team_1_possession"] + possession["team_2_possession"]
    assert shares == pytest.approx(1.0) or shares == 0.0
    assert 0.0 <= possession["coverage"] <= 1.0
    assert possession["total_frames"] == N_FRAMES


def test_no_impossible_speeds_end_to_end(tmp_path, clip, stub, config):
    report = run_pipeline(tmp_path, clip, stub, config)
    assert all(row["top_speed_kmh"] <= 36.0 for row in report["tracks"])


def test_track_rows_carry_span_and_coverage(tmp_path, clip, stub, config):
    report = run_pipeline(tmp_path, clip, stub, config)
    assert report["tracks"], "no tracks summarised"
    for row in report["tracks"]:
        assert {"track_id", "span_seconds", "coverage", "distance_m"} <= set(row)


def test_max_frames_limits_the_run(tmp_path, clip, stub, config):
    args = main.parse_args(
        [
            "--input", str(clip),
            "--output", str(tmp_path / "out.mp4"),
            "--config", str(config),
            "--track-stub", str(stub),
            "--use-stub",
            "--max-frames", "20",
        ]
    )
    assert main.run(args)["frames_analysed"] == 20


def test_no_video_flag_skips_rendering(tmp_path, clip, stub, config):
    args = main.parse_args(
        [
            "--input", str(clip),
            "--output", str(tmp_path / "out.mp4"),
            "--config", str(config),
            "--track-stub", str(stub),
            "--use-stub",
            "--no-video",
        ]
    )
    main.run(args)
    assert not (tmp_path / "out.mp4").exists()


def test_mismatched_stub_is_refused(tmp_path, clip, stub, config):
    other = tmp_path / "other.mp4"
    save_video([np.zeros((HEIGHT, WIDTH, 3), np.uint8)] * 4, other, fps=FPS)

    args = main.parse_args(
        [
            "--input", str(other),
            "--output", str(tmp_path / "out.mp4"),
            "--config", str(config),
            "--track-stub", str(stub),
            "--use-stub",
        ]
    )
    with pytest.raises(ValueError, match="does not match this run"):
        main.run(args)
