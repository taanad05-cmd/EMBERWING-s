# EMBERWING

A 43-second animated film made entirely in code by **Claude (model: `claude-opus-5-5`)** in a single
Claude session. The prompt is in [`PROMPT.md`](PROMPT.md).

**Watch:** [`films/emberwing.mp4`](films/emberwing.mp4), or download
[`films/emberwing.html`](films/emberwing.html) and open it in a browser (the whole film in one
self-contained, full-screen file).

No image models, no video models, no game engine, no stock footage, no sound samples. Everything
is generated from Python code:

- **A 3D world renderer written from scratch** (`src/world.py`): a procedural 2048×2048 tiling
  heightfield with a carved river canyon, baked sun shadows and ambient occlusion, rock/grass/
  forest/snow materials, a per-pixel raymarcher compiled with numba, water with rippled
  reflections, height fog plus valley mist, a sunrise sky with stars, a cloud layer, and a
  sea of clouds.
- **Cinematic post** (`src/film3.py`): bloom, god rays, ACES tone mapping, split-tone grading,
  depth of field, vignette and grain.
- **The dragon** (`src/dragon.py`): a parametric 2D character rig with cel shading and ink lines,
  composited into the 3D scenes.
- **The score and sound** (`src/audio3.py` on top of `src/audio.py`): synthesized strings, choir,
  brass, piano, bells, taiko and timpani, plus wind, wing beats, risers and impacts.

## The film

1. Dawn: stars fade over a vast mountain range.
2. The little dragon on a peak, sunrise in its eyes.
3. It dives off the cliff; its wings burst open.
4. A high-speed chase low through a river canyon.
5. It punches up through the clouds into a sea of clouds.
6. Spirit dragons rise; its wings ignite with embers.
7. Its silhouette crosses the sun: EMBERWING.

## Run it

```bash
pip install numpy scipy pillow numba
# needs ffmpeg and Noto Serif CJK fonts
cd src
python3 render3.py      # renders seg3/*.mp4 (about 35 min on 2 CPU cores)
python3 audio3.py       # -> score3.wav
```

Join `seg3/S1.mp4 … S7.mp4` in shot order with ffmpeg's concat demuxer and add `score3.wav`.
