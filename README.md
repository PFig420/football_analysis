# Football Analysis Project

Detect and track players, referees and the ball in match footage with YOLO, assign players
to teams from kit colour, and measure possession, camera movement, player speed and distance
covered.

Originally Abdullah Tarek's tutorial project; see `FIXES.md` for what was rebuilt and why.

![Screenshot](output_videos/screenshot.png)

## What it does

- **Detection and tracking** — YOLO for detection, ultralytics' ByteTrack for identity.
  Players, goalkeepers, referees and the ball are kept as separate classes.
- **Team assignment** — k-means over shirt pixels, decided by majority vote across each
  track's first several appearances. Goalkeepers are placed by pitch position instead, since
  their kit matches neither team.
- **Camera movement** — every frame is registered into a reference view with a full
  homography (SIFT + RANSAC), so pan, tilt and zoom are all accounted for. A cheaper
  translation-only model is available for static cameras.
- **Perspective transform** — pixel positions to pitch metres, so speed is in km/h and
  distance in metres rather than pixels.
- **Possession** — nearest player to the ball, counted only on frames where the ball was
  actually detected.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# CPU-only torch is a much smaller download:
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Detection and tracking both come from ultralytics. `supervision`, which the original used
for ByteTrack, is no longer a dependency — it removed `ByteTrack` in 0.31.0, and ultralytics
ships the same algorithm.

## Assets

Neither the weights nor the sample clip are in the repository.

- [Trained YOLO v5 weights](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing) → `models/best.pt`
- [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing) → `input_videos/`

## Run

```bash
python main.py --input input_videos/08fd33_4.mp4 --output output_videos/out.mp4
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--config config.yaml` | override any default in `config.py` |
| `--track-stub stubs/tracks.pkl --use-stub` | cache detections and reuse them |
| `--report report.json` | per-track distances, speeds, coverage and possession |
| `--no-video` | measure only, skip rendering |
| `--max-frames N` | process the first N frames |

### Camera model

`camera.model` is `homography` (default) or `translation`. Use the default unless the camera
is genuinely fixed. Measured on the sample clip, where the camera pans about 860 px to follow
play, the homography places static scenery to a median **0.68 px** while the best possible
translation is off by **60.80 px** — an error far larger than the player movement being
measured. The homography costs about 28 s per 750 frames of 1080p.

Detection is the slow part. Cache it once with `--track-stub`, then iterate on the analysis
with `--use-stub`. The stub records which video and model produced it and is refused if
either differs, so a stale cache cannot silently mask your changes.

## Calibrate the pitch before trusting the numbers

**Speed and distance are only as good as the perspective quad.** The built-in default was
eyeballed by the original author from one clip; it is marked `calibrated: false` and the
pipeline warns on every run until you replace it.

```bash
python calibrate_pitch.py --input input_videos/clip.mp4 --out config.yaml
python main.py --input input_videos/clip.mp4 --config config.yaml
```

This fits a homography to the markings it can find and writes the `pitch:` block for you,
along with an overlay image that redraws the pitch from world coordinates so you can check
the fit by eye. **Look at the overlay**: the drawn touchlines, halfway line and centre circle
should sit on the real ones.

What has to be in the frame:

* the **halfway line**, end to end,
* both **touchlines**,
* the **centre circle**, at least in part.

No penalty area is needed, which matters because a camera covering the middle third never
sees one. Pitch width is assumed to be 68 m; pass `--width` if yours differs.

Calibrate on **frame 0**. The camera model maps every other frame into it, so that is the
coordinate system the quad lives in.

The run log reports what fraction of player positions fell inside the quad; anything outside
gets no speed or distance. On the sample clip, calibrating took coverage from 58.3 % to
71.4 %.

## Tests

```bash
pytest
```

109 tests that need nothing but the repository — the committed 750-frame stub in `stubs/`
stands in for detection, and the camera model is verified against synthetic pans with known
answers.

A further 11 run only when the sample clip is present, and check the camera model against
the real thing: it must place static scenery to within 1.5 px. Deselect them with
`pytest -m "not slow"`.

## Training

`training/football_training_yolo_v5.ipynb` fine-tunes YOLO on the Roboflow
football-players-detection dataset. Supply your own Roboflow API key through the environment;
the one in the notebook's history is revoked.

## Project layout

| Path | Purpose |
| --- | --- |
| `main.py` | CLI and the two-pass pipeline |
| `config.py` | every tunable, with defaults |
| `trackers/` | YOLO + ByteTrack, ball interpolation, annotation |
| `team_assigner/` | kit-colour clustering, goalkeeper placement |
| `camera_movement_estimator/` | optical flow, RANSAC global motion |
| `view_transformer/` | pixel → pitch-metre homography |
| `speed_and_distance_estimator/` | per-track kinematics |
| `player_ball_assigner/` | possession |
| `calibrate_pitch.py` | fit the pitch homography from visible markings |
| `tests/` | pytest suite |
| `FIXES.md` | what was wrong, what changed, what is still open |
