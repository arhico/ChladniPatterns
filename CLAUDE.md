# ChladniPatterns

Single-file interactive Chladni plate simulator: `ChladniPatterns.py` (numpy + matplotlib + PySide6).

## Run
- `python ChladniPatterns.py` (deps in `requirements.txt`; venv in `.venv/` if present)
- Headless smoke test: `QT_QPA_PLATFORM=offscreen timeout 20 python -c "import ChladniPatterns as C; ..."`, drive it with `QtCore.QTimer.singleShot` and `c.qapp.quit()`.
- Video export needs `ffmpeg` on PATH (falls back to GIF via Pillow).
- Detailed run/test/profile/extend recipes: `.claude/skills/chladni-patterns/SKILL.md`.
- Renders replay LFOs from phase 0 via `apply_lfos`; new sliders need a `key` + entry in `_state()`/`params_of()`.

## Architecture
- `Plate`: Kirchhoff mode frequencies + cosine mode shapes; `field()` / `spectrum()` fan out over `POOL` (ThreadPoolExecutor — numpy releases the GIL).
- `Chladni._physics_loop` (worker thread, `SAND_RATE` Hz): on a `_version` change computes ONE field at `_render_size` (screen px × Quality), `upscale`s it (two lerp-matrix matmuls) to `_plate_px`; sand physics and colouring both use that full-res amp. Steps sand unless `paused`; builds the frame with `compose()`.
- `Plate.energy` caches unit-amplitude |u|² per generator (real float32 BLAS; complex matmul is ~30× slower), so amplitude edits only re-weight. Spectrum is recomputed only when plate/drive/damping change.
- Images are (h, w) uint32 RGBA. Worker caches the coloured field (`field_rgba`) and per frame only blends sandy pixels (`compose`).
- No `imshow` for the plate: `paste()` copies the frame 1:1 into the Agg buffer at `plate_rect(ax)` (whole device pixels), then spines + exciter mark are drawn on top. matplotlib resampling cost ~35 ms/frame and left edge seams.
- GUI thread (`_tick`, every `FRAME_MS`) steps the per-slider LFOs, then blits the plate and spectrum axes. Never call `draw_idle` per change (full figure redraw = lag) except for static-artist changes (title/limits); never add many-point artists like `scatter`.
- Shared state: whole-object swaps of `_result`, `_frame`, `_sand`; parameter edits go through `_set_attr` / `_set_item` under `_lock` and bump `_version`.
- `OPENBLAS_NUM_THREADS=1` is set before numpy import (POOL parallelises; BLAS threads oversubscribe). `_shutdown` joins threads on quit — skipping it crashes at interpreter exit.
- `_export` re-simulates offline on a private Agg figure, piping RGBA frames to ffmpeg.

## Conventions
- Keep it one file; match the existing terse style and section comments (`# ---- ...`).
- Update the module docstring "Controls" list when adding UI controls.
- `renders/` holds user output (videos, screenshots): never delete the folder or files you didn't create.
