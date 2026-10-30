import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def track_stub() -> dict:
    """The committed 750-frame fixture from the sample clip.

    Everything downstream of YOLO is testable from this alone -- no weights, no video.
    """
    path = ROOT / "stubs" / "track_stubs.pkl"
    if not path.exists():
        pytest.skip("stubs/track_stubs.pkl is missing")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def textured_image() -> np.ndarray:
    """A deterministic, corner-rich image for optical-flow tests."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(480, 640), dtype=np.uint8)
    # Blur-free random noise is too easy to track; add structure at a few scales so the
    # corner detector behaves the way it does on real footage.
    image = np.zeros((480, 640), dtype=np.uint8)
    for block in (4, 8, 16):
        small = base[::block, ::block]
        upscaled = np.kron(small, np.ones((block, block), dtype=np.uint8))
        image = np.maximum(image, upscaled[:480, :640] // 3)
    return image
