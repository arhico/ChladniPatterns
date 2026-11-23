---
name: chladni-patterns
description: Run, test, profile and extend the Chladni plate simulator (ChladniPatterns.py). Use when launching the app, smoke-testing a change headlessly, adding a control/LFO target, touching rendering/export, or chasing lag.
---

# Chladni Patterns

## Launch
- GUI: `.venv/bin/python ChladniPatterns.py` (fall back to `python` if no `.venv`).
- Headless check (no display needed), drive with timers and quit:
  ```sh
  QT_QPA_PLATFORM=offscreen timeout 30 .venv/bin/python -c "
  import ChladniPatterns as C, sys
  from matplotlib.backends.qt_compat import QtCore
  c = C.Chladni(); c.window.show()
  sys.excepthook = lambda *a: (print('EXC', a), c.qapp.quit())
  QtCore.QTimer.singleShot(1500, lambda: (print(c._render_size, c._shown_frame), c.qapp.quit()))
  c.qapp.exec()" 2>&1 | grep -v propagate
  ```
- Useful handles: `c.f_controls[i]`, `c.a_controls[i]`, `c.controls` (all ValueControls), `ctl.lfo.setChecked(True)`, `ctl.lfo_rate/lfo_depth`, `c.btn_pause`, `c._screenshot()`, `c.spin_duration.setValue(1); c._start_export()` then poll `c._export_status`.
- FPS: compare `c._shown_frame` across a timer interval (idle cap is `SAND_RATE` = 60).
- Screenshot/video output lands in `renders/`. Delete only files your test created; never the folder or the user's files.

## Profiling
Time pieces directly, without the GUI:
```sh
.venv/bin/python -c "
import ChladniPatterns as C, numpy as np, time
p = C.Plate(.3, .3, .001, 'Steel')
t = time.perf_counter(); p.field(((800, 1, True),), (.5, .5), .02, 513, 513); print(time.perf_counter() - t)"
```
Check `uptime` first: another running instance of the app skews the numbers.

## Extending
- **New slider**: `ValueControl(grid, row, label, lo, hi, value, step, decimals, color, on_change)`. Append it to `self.controls` (so its LFO ticks) and give it a `key` so renders replay its LFO. Keys are `(state_name, index_or_None)`, handled by `apply_lfos`. Also add the value to `_state()` and, if it affects physics, to `params_of()`.
- Physics parameters must go through `_set_attr` / `_set_item` (they bump `_version` under `_lock`); otherwise the worker won't recompute.
- Document new controls in the module docstring "Controls" section and the README table.

## Performance rules (the causes of past lag)
- No per-change `canvas.draw_idle()`. Plate and spectrum artists are `animated=True` and blitted in `_tick`. A full redraw is OK only for static artists (title/limits, `_refresh_geometry`).
- No many-point matplotlib artists (`scatter`), and no `imshow` for the plate: frames are `paste`d into the Agg buffer pixel-exact.
- Never resize RGBA per frame; resample the scalar field (`upscale`, matmul-based) once per change, then colour it.
- Field maths stays in real float32 BLAS (`Plate.energy`). Complex `@` is ~30× slower. Per-generator energy is cached, so amplitude changes are only a reweighting.
- Don't submit to `POOL` from inside a `POOL` task (nested waits can deadlock).
