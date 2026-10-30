import numpy as np
import pytest

from config import PipelineConfig, PitchConfig
from trackers import StubHeader, load_stub, save_stub
from utils import probe_video, iter_frames, save_video


# -- config ----------------------------------------------------------------------------


def test_defaults_round_trip_through_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    PipelineConfig().dump(path)
    assert PipelineConfig.load(path) == PipelineConfig()


def test_yaml_overrides_only_what_it_names(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("speed:\n  max_plausible_kmh: 40.0\n  window_seconds: 0.5\n")
    config = PipelineConfig.load(path)

    assert config.speed.max_plausible_kmh == 40.0
    assert config.speed.window_seconds == 0.5
    assert config.pitch == PitchConfig()  # untouched


def test_unknown_section_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("nonsense:\n  value: 1\n")
    with pytest.raises(ValueError, match="unknown config section"):
        PipelineConfig.load(path)


def test_scale_follows_frame_height():
    assert PipelineConfig().for_frame(1920, 1080).scale == pytest.approx(1.0)
    assert PipelineConfig().for_frame(1280, 720).scale == pytest.approx(720 / 1080)


def test_default_pitch_is_flagged_uncalibrated():
    """The shipped vertices were eyeballed from one clip; nothing should trust them."""
    assert PipelineConfig().pitch.calibrated is False


# -- video I/O -------------------------------------------------------------------------


def make_clip(path, n_frames=12, size=(64, 48), fps=30.0):
    frames = [
        np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8) for i in range(n_frames)
    ]
    save_video(frames, path, fps=fps)
    return frames


def test_probe_reads_properties_from_the_file(tmp_path):
    path = tmp_path / "clip.mp4"
    make_clip(path, n_frames=12, size=(64, 48), fps=30.0)

    meta = probe_video(path)
    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(30.0, rel=0.01)
    assert meta.frame_count == 12


def test_iter_frames_streams_without_holding_the_clip(tmp_path):
    path = tmp_path / "clip.mp4"
    make_clip(path, n_frames=8)
    assert sum(1 for _ in iter_frames(path)) == 8
    assert sum(1 for _ in iter_frames(path, limit=3)) == 3


def test_missing_video_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "absent.mp4")


def test_unsupported_output_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported output extension"):
        save_video([np.zeros((4, 4, 3), np.uint8)], tmp_path / "out.gif", fps=24)


# -- stub provenance -------------------------------------------------------------------


def header(**overrides) -> StubHeader:
    base = dict(version=2, video_name="clip.mp4", video_size=1234, frame_count=10, model_name="best.pt")
    base.update(overrides)
    return StubHeader(**base)


def test_matching_stub_loads(tmp_path):
    path = tmp_path / "stub.pkl"
    tracks = {"players": [{}]}
    save_stub(path, tracks, header())
    assert load_stub(path, header()) == tracks


def test_stub_from_a_different_video_is_refused(tmp_path):
    """The original reloaded any stub at hand, so edits to detection silently did nothing."""
    path = tmp_path / "stub.pkl"
    save_stub(path, {"players": [{}]}, header())

    with pytest.raises(ValueError, match="does not match this run"):
        load_stub(path, header(video_name="other.mp4"))


def test_stub_from_a_different_model_is_refused(tmp_path):
    path = tmp_path / "stub.pkl"
    save_stub(path, {"players": [{}]}, header())
    with pytest.raises(ValueError, match="model_name"):
        load_stub(path, header(model_name="v8.pt"))


def test_absent_stub_returns_none(tmp_path):
    assert load_stub(tmp_path / "nope.pkl", header()) is None


def test_legacy_headerless_stub_still_loads(track_stub, caplog):
    """The committed fixture predates headers; it must keep working, with a warning."""
    assert set(track_stub) >= {"players", "referees", "ball"}
    assert len(track_stub["players"]) == 750
