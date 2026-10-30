"""Resolution-independent drawing helpers.

Every overlay in the original was positioned with absolute pixel coordinates tuned for
1920x1080 -- the possession panel at (1350, 850)-(1900, 970), the camera panel at
(0, 0)-(500, 100) -- so on any other clip they landed off-frame or on top of the action.
Here a panel is given as fractions of the frame and font sizes scale with its height.
"""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def fractional_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) fractions into pixel coordinates for this frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))


def draw_text_panel(
    frame: np.ndarray,
    lines: list[str],
    box: tuple[float, float, float, float],
    scale: float = 1.0,
    alpha: float = 0.55,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
) -> np.ndarray:
    """Draw a translucent panel with one line of text per entry, sized to the panel."""
    if not lines:
        return frame
    x1, y1, x2, y2 = fractional_box(frame, box)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    panel_height = max(y2 - y1, 1)
    line_height = panel_height / (len(lines) + 0.5)
    font_scale = max(0.35, min(1.2, line_height / 32.0))
    thickness = max(1, int(round(2 * font_scale)))
    pad = max(4, int(0.02 * (x2 - x1)))

    for i, line in enumerate(lines):
        baseline = int(y1 + line_height * (i + 1))
        cv2.putText(frame, line, (x1 + pad, baseline), FONT, font_scale, fg, thickness, cv2.LINE_AA)
    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 1.0,
    color=(0, 0, 0),
    outline=(255, 255, 255),
) -> np.ndarray:
    """Small text with a light outline, so it stays readable over grass or kit."""
    font_scale = 0.5 * scale
    thickness = max(1, int(round(1.5 * scale)))
    cv2.putText(frame, text, origin, FONT, font_scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, font_scale, color, thickness, cv2.LINE_AA)
    return frame
