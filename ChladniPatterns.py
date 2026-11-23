"""Interactive Chladni plate simulator.

A rectangular plate of real dimensions (width x height x thickness, in a chosen
material) is driven at one point by up to four sine generators. Each generator
excites every plate mode (m, n) with the steady-state response of a damped
oscillator; because the generators run at different frequencies, their
time-averaged vibration energies add. Sand grains get random kicks proportional
to the local vibration amplitude, so they drift out of the antinodes and pile
up on the nodal lines - just like a real plate.

Mode frequencies follow Kirchhoff plate theory,
    f_mn = (pi / 2) * sqrt(D / (rho h)) * ((m / Lx)^2 + (n / Ly)^2),
    D    = E h^3 / (12 (1 - nu^2)),
with cosine (free-edge-like) mode shapes.

Physics (field, spectrum, sand) and compositing of the plate image run off the
GUI thread: per-generator fields and the spectrum are computed in parallel in a
thread pool (numpy releases the GIL), and the sand is splatted into the field
image with numpy, so the GUI thread only blits one ready-made RGB image. Controls are native Qt widgets beside the plot.

Controls
    sliders        frequency / amplitude per generator, plate size / thickness /
                   damping, exciter x / y position, shake strength
    ~ buttons      per-slider LFO: click toggles it, the arrow sets rate / depth;
                   the slider (or typed value) sets the centre it swings around
    number boxes   type an exact value (applied on Enter or when leaving the box);
                   arrow keys / mouse wheel step it
    check boxes    enable generators (the coloured f labels), toggle field / sand
    Material       plate material
    click plate    move the exciter (updates the x / y controls; breaks symmetry)
    Reset sand     scatter the sand uniformly again
    Colormap       colour scheme of the field (also used in screenshots / videos)
    Quality        field resolution relative to the plate's on-screen pixel size
                   (upscaled bilinearly; sand is always drawn at full resolution)
    Pause          freeze the sand (the field still follows parameter edits)
    Screenshot     save the current plot area as a PNG to renders/
    Cross in captures   draw the exciter cross in screenshots and videos
    Render video   simulate the current setup for the chosen length (starting
                   from the current sand) and save it to renders/; resumes a
                   paused simulation and retriggers all LFOs (phase 0) both live
                   and in the video, so what you see is what gets recorded
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # we parallelise ourselves (POOL)
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.style
from matplotlib.backend_bases import scroll_handler
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

GRID = 360          # default field resolution before the plate's screen size is known
QUALITY = {"Full": 1.0, "1/2": 0.5, "1/4": 0.25}   # display resolution vs screen pixels
F_MAX = 4000.0      # Hz, top of the generator / spectrum range
N_SAND = 20000
SAND_RATE = 60.0    # sand steps per second (physics thread)
FRAME_MS = 20       # GUI refresh interval
MAX_MODES = 48      # cap on mode indices per direction
VIDEO_FPS = 30
VIDEO_DIR = Path(__file__).resolve().parent / "renders"

# Young's modulus [Pa], density [kg/m^3], Poisson ratio
MATERIALS = {
    "Steel": (200e9, 7850.0, 0.30),
    "Aluminium": (69e9, 2700.0, 0.33),
    "Brass": (100e9, 8500.0, 0.34),
    "Glass": (70e9, 2500.0, 0.22),
}

BG = "#121212"
PANEL = "#1e1e1e"
FG = "#d8d8d8"
MUTED = "#555555"
MUTED_TEXT = "#8a8a8a"
GREY = "#9e9e9e"
CYAN = "#00e5ff"
GEN_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373"]

POOL = ThreadPoolExecutor(max_workers=6)
# Images are (ny, nx) uint32 holding RGBA bytes (alpha 255), so colouring is one
# 4-byte gather and pasting into the Agg buffer is a plain copy.
COLORMAPS = ["inferno", "magma", "plasma", "viridis", "cividis", "turbo",
             "twilight", "coolwarm", "ocean", "bone", "gray"]
SAND_RGB = np.array(mpl.colors.to_rgb("#f3e3c0"), np.float32) * 255
EMPTY_RGBA = np.array([*np.round(np.array(mpl.colors.to_rgb("#0b0b0b")) * 255), 255],
                      np.uint8).view(np.uint32)[0]


def plate_speed(h, material):
    """sqrt(D / (rho h)) [m^2/s] of a plate of thickness h [m]."""
    E, rho, nu = MATERIALS[material]
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    return np.sqrt(D / (rho * h))


def mode_count(f1):
    """Mode indices needed in one direction to cover the frequency range."""
    return int(np.clip(np.ceil(np.sqrt(2.0 * F_MAX / f1)) + 1, 4, MAX_MODES))


class Plate:
    """Mode shapes and frequencies of an lx x ly x h plate [m]."""

    def __init__(self, lx, ly, h, material):
        c = plate_speed(h, material)
        self.fx = np.pi / 2 * c / lx ** 2            # frequency of mode (1, 0)
        self.fy = np.pi / 2 * c / ly ** 2            # frequency of mode (0, 1)
        self.kx = np.arange(mode_count(self.fx))
        self.ky = np.arange(mode_count(self.fy))
        self._basis = {}
        self._energy = {}                            # unit-amplitude |u|^2 per generator setup
        self._energy_lock = threading.Lock()         # filled from pool threads
        self.f = self.fy * self.ky[:, None] ** 2 + self.fx * self.kx[None, :] ** 2   # [n_y, m_x]
        self.active = self.f > 0                     # drop the rigid (0,0) mode
        self.audible = self.active & (self.f < 2.0 * F_MAX)

    def basis(self, k, n):
        """cos(pi k x) sampled at n points, cached per resolution."""
        key = (len(k), n)
        if key not in self._basis:
            self._basis[key] = np.cos(np.pi * k[:, None] * np.linspace(0.0, 1.0, n)[None, :]).astype(np.float32)
        return self._basis[key]

    def drive_weights(self, drive):
        dx, dy = drive
        return np.outer(np.cos(np.pi * self.ky * dy), np.cos(np.pi * self.kx * dx))

    def transfer(self, f, zeta):
        """Damped-oscillator response of every mode to a sine at frequency f."""
        h = np.zeros_like(self.f, dtype=complex)
        fm = self.f[self.active]
        h[self.active] = 1.0 / (fm ** 2 - f ** 2 + 2j * zeta * fm * f)
        return h

    def field(self, gens, drive, zeta, nx=GRID, ny=GRID):
        """Normalised vibration amplitude sampled on an (ny, nx) grid."""
        return self.fields(gens, drive, zeta, [(nx, ny)])[0]

    def fields(self, gens, drive, zeta, sizes):
        """field() at several (nx, ny) resolutions, all computed in parallel."""
        active = [(f, a) for f, a, on in gens if on and a > 0]
        jobs = [[POOL.submit(self.energy, f, drive, zeta, nx, ny) for f, _ in active]
                for nx, ny in sizes]
        out = []
        for (nx, ny), row in zip(sizes, jobs):
            amp = np.sqrt(sum((a * a * j.result() for (_, a), j in zip(active, row)),
                              np.zeros((ny, nx), np.float32)))
            peak = amp.max()
            out.append(amp / peak if peak > 0 else amp)
        return out

    def energy(self, f, drive, zeta, nx, ny):
        """|u|^2 for a unit-amplitude generator at f; cached, so amplitude edits are cheap."""
        key = (f, drive, zeta, nx, ny)
        e = self._energy.get(key)
        if e is None:
            c = self.drive_weights(drive) * self.transfer(f, zeta)
            by_t, bx = self.basis(self.ky, ny).T, self.basis(self.kx, nx)
            re = by_t @ c.real.astype(np.float32) @ bx
            im = by_t @ c.imag.astype(np.float32) @ bx         # real BLAS: ~30x faster than complex
            e = re * re + im * im
            with self._energy_lock:
                if len(self._energy) >= 32:
                    self._energy.pop(next(iter(self._energy)))
                self._energy[key] = e
        return e

    def spectrum(self, f, drive, zeta):
        """Drive-point response vs frequency, to show where the resonances are."""
        w = self.drive_weights(drive)[self.audible] ** 2
        fm = self.f[self.audible]

        def chunk(fc):
            h = 1.0 / (fm[None, :] ** 2 - fc[:, None] ** 2 + 2j * zeta * fm[None, :] * fc[:, None])
            return (w[None, :] * np.abs(h) ** 2).sum(axis=1)

        parts = POOL.map(chunk, np.array_split(f, 4))
        return np.concatenate(list(parts)) + 1e-30


def apply_dark_theme(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, color in ((roles.Window, BG), (roles.Base, "#1a1a1a"), (roles.AlternateBase, PANEL),
                        (roles.Button, "#2c2c2c"), (roles.Text, FG), (roles.WindowText, FG),
                        (roles.ButtonText, FG), (roles.ToolTipBase, PANEL), (roles.ToolTipText, FG),
                        (roles.Highlight, "#3d6fa8"), (roles.HighlightedText, "#ffffff"),
                        (roles.PlaceholderText, MUTED_TEXT)):
        pal.setColor(role, QtGui.QColor(color))
    pal.setColor(QtGui.QPalette.ColorGroup.Disabled, roles.Text, QtGui.QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QGroupBox {{ border: 1px solid #333; border-radius: 4px; margin-top: 1.1em; padding-top: 4px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {MUTED_TEXT}; }}
        QCheckBox::indicator {{ width: 11px; height: 11px; border: 1px solid #777; border-radius: 2px;
                                background: #1a1a1a; }}
        QCheckBox::indicator:checked {{ background: #cfd8dc; border-color: #cfd8dc; }}
        QSlider::groove:horizontal {{ height: 4px; background: #2c2c2c; border-radius: 2px; }}
        QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: #e0e0e0; }}
    """)


class ValueControl:
    """Slider + spin box for one float value, laid out on one grid row.

    The slider moves in multiples of *step*; the spin box accepts any typed
    value (to *decimals* places) and applies it on Enter / focus loss.
    """

    def __init__(self, grid, row, label, lo, hi, value, step, decimals, color, on_change):
        self.step = step
        self.on_change = on_change
        self.key = None                 # state key the LFO drives in renders (see apply_lfos)
        self.center = value             # LFO swings around this; follows user edits
        self.phase = 0.0
        self._modulating = False
        if isinstance(label, str):
            label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setStyleSheet(f"QSlider::sub-page:horizontal {{ background: {color}; border-radius: 2px; }}")
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)        # don't apply half-typed numbers
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.spin.setMinimumWidth(95)
        self.set_range(lo, hi)
        self.set_value(value, notify=False)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        grid.addWidget(label, row, 0)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.spin, row, 2)
        grid.addWidget(self._build_lfo(color), row, 3)

    def _build_lfo(self, color):
        self.lfo = QtWidgets.QToolButton()
        self.lfo.setText("~")
        self.lfo.setCheckable(True)
        self.lfo.setToolTip("LFO on/off (arrow: rate / depth)")
        self.lfo.setStyleSheet(f"QToolButton:checked {{ color: {color}; font-weight: bold; }}")
        self.lfo.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.lfo.toggled.connect(self._lfo_toggled)
        form = QtWidgets.QWidget()
        lay = QtWidgets.QFormLayout(form)
        self.lfo_rate = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.01, maximum=20,
                                                 value=0.2, singleStep=0.05, suffix=" Hz")
        self.lfo_depth = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0, maximum=100,
                                                  value=10, singleStep=1, suffix=" %")
        self.lfo_depth.setToolTip("Swing ± this share of the slider range")
        lay.addRow("Rate", self.lfo_rate)
        lay.addRow("Depth", self.lfo_depth)
        menu = QtWidgets.QMenu(self.lfo)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(form)
        menu.addAction(action)
        self.lfo.setMenu(menu)
        return self.lfo

    def _lfo_toggled(self, on):
        self.phase = 0.0
        if not on:
            self.set_value(self.center)             # come back to rest

    def lfo_spec(self):
        """Frozen LFO settings (key, centre, swing, rate, lo, hi) for apply_lfos, or None."""
        if not self.lfo.isChecked():
            return None
        hi = self.spin.maximum()
        return (self.key, self.center, self.lfo_depth.value() / 100 * (hi - self.lo),
                self.lfo_rate.value(), self.lo, hi)

    def lfo_step(self, dt):
        """Advance the LFO by dt seconds and apply the modulated value."""
        if not self.lfo.isChecked():
            return
        self.phase = (self.phase + 2 * np.pi * self.lfo_rate.value() * dt) % (2 * np.pi)
        swing = self.lfo_depth.value() / 100 * (self.spin.maximum() - self.lo)
        value = float(np.clip(self.center + swing * np.sin(self.phase), self.lo, self.spin.maximum()))
        self._modulating = True
        try:
            self.set_value(value)
        finally:
            self._modulating = False

    @property
    def value(self):
        return self.spin.value()

    def set_range(self, lo, hi):
        self.lo = lo
        value = self.spin.value()
        with QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider):
            self.spin.setRange(lo, hi)
            self.slider.setRange(0, max(1, round((hi - lo) / self.step)))
            self.spin.setValue(value)
            self.slider.setValue(round((self.spin.value() - lo) / self.step))

    def set_value(self, value, notify=True):
        if notify:
            self.spin.setValue(value)               # emits -> _from_spin -> on_change
            return
        with QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(round((self.spin.value() - self.lo) / self.step))
        if not self._modulating:
            self.center = self.spin.value()

    def _from_slider(self, pos):
        with QtCore.QSignalBlocker(self.spin):
            self.spin.setValue(self.lo + pos * self.step)
        self._from_spin(self.spin.value(), move_slider=False)

    def _from_spin(self, value, move_slider=True):
        if move_slider:
            with QtCore.QSignalBlocker(self.slider):
                self.slider.setValue(round((value - self.lo) / self.step))
        if not self._modulating:
            self.center = value
        self.on_change(value)


def lfo_value(spec, t):
    """Value of an LFO (from ValueControl.lfo_spec) t seconds after a retrigger."""
    _, center, swing, rate, lo, hi = spec
    return float(np.clip(center + swing * np.sin(2 * np.pi * rate * t), lo, hi))


def apply_lfos(state, specs, t):
    """Copy of *state* (Chladni._state) with every LFO applied at time t."""
    st = dict(state, freqs=list(state["freqs"]), amps=list(state["amps"]))
    drive = list(st["drive"])
    for spec in specs:
        (name, i), v = spec[0], lfo_value(spec, t)
        if name == "drive_cm":
            drive[i] = v / (st["width_cm"], st["height_cm"])[i]
        elif i is None:
            st[name] = v
        else:
            st[name][i] = v
    st["drive"] = (min(drive[0], 1.0), min(drive[1], 1.0))
    return st


def params_of(st):
    """Physics parameters from a state snapshot."""
    return dict(
        gens=tuple(zip(st["freqs"], st["amps"], st["on"])),
        drive=st["drive"],
        zeta=st["zeta"],
        plate=(st["width_cm"] / 100, st["height_cm"] / 100, st["thick_mm"] / 1000, st["material"]),
    )


def sand_pixels(sand, nx=GRID, ny=GRID):
    """(N, 2) grid indices (column, row) of the grains."""
    size = np.array([nx, ny])
    return np.clip(np.rint(sand * (size - 1)).astype(np.intp), 0, size - 1)


@lru_cache(maxsize=None)
def field_lut(cmap):
    """256-entry RGBA lookup table (as uint32) for a matplotlib colormap."""
    rgba = np.round(mpl.colormaps[cmap](np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)
    return rgba.view(np.uint32).ravel()


def field_rgba(amp, cmap):
    """Field coloured with *cmap* (None hides it); only changes with the parameters, so cache it."""
    if cmap:
        return field_lut(cmap)[np.clip(amp * 255, 0, 255).astype(np.uint8)]
    return np.full(amp.shape, EMPTY_RGBA)


def compose(base, sand, show_sand):
    """*base* (field_rgba) with the sand splatted on top; only sandy pixels are blended."""
    rgba = base.copy()
    if show_sand:
        ny, nx = base.shape
        idx = sand_pixels(sand, nx, ny)
        pix, counts = np.unique(idx[:, 1] * nx + idx[:, 0], return_counts=True)
        alpha = np.minimum(counts * np.float32(0.6), 1.0)[:, None]
        flat = rgba.reshape(-1).view(np.uint8).reshape(-1, 4)
        under = flat[pix, :3].astype(np.float32)
        flat[pix, :3] = under + (SAND_RGB - under) * alpha
    return rgba


@lru_cache(maxsize=8)
def lerp_matrix(n_out, n_in):
    """(n_out, n_in) linear-interpolation matrix: M @ v resamples v to n_out points."""
    pos = np.linspace(0, n_in - 1, n_out)
    i0 = np.minimum(pos.astype(np.intp), n_in - 2) if n_in > 1 else np.zeros(n_out, np.intp)
    w = (pos - i0).astype(np.float32)
    m = np.zeros((n_out, n_in), np.float32)
    rows = np.arange(n_out)
    m[rows, i0] = 1 - w
    if n_in > 1:
        m[rows, i0 + 1] += w
    return m


def upscale(amp, w, h):
    """Bilinear resize of a field to (h, w) as two BLAS matmuls (separable + linear)."""
    sh, sw = amp.shape
    if (sw, sh) == (w, h):
        return amp
    return lerp_matrix(h, sh) @ amp @ lerp_matrix(w, sw).T


def plate_rect(ax):
    """Whole device pixels (x0, y0, w, h) inside the axes, y measured from the bottom."""
    b = ax.bbox
    x0, y0 = int(np.ceil(b.x0)), int(np.ceil(b.y0))
    return x0, y0, max(8, int(b.x1) - x0), max(8, int(b.y1) - y0)


def paste(canvas, rect, rgba):
    """Copy *rgba* (origin bottom-left) 1:1 into the Agg buffer: no matplotlib resampling."""
    buf = np.asarray(canvas.buffer_rgba())
    x0, y0, w, h = rect
    top = buf.shape[0] - y0 - h
    if rgba.shape != (h, w) or top < 0 or x0 + w > buf.shape[1]:
        return                                    # stale frame from before a resize
    buf.view(np.uint32)[top:top + h, x0:x0 + w, 0] = rgba[::-1]


def step_sand(sand, amp, step_scale, rng):
    """One random-walk step; returns a new array (the GUI may hold the old one)."""
    idx = sand_pixels(sand, amp.shape[1], amp.shape[0])
    a = amp[idx[:, 1], idx[:, 0]]
    step = 0.01 * (a + 0.002)
    sand = sand + rng.standard_normal(sand.shape) * step[:, None] * step_scale
    sand = 1.0 - np.abs(1.0 - np.abs(sand))       # reflect at edges
    return np.clip(sand, 0.0, 1.0, out=sand)


class Chladni:
    def __init__(self):
        self.freqs = [800.0, 1600.0, 2600.0, 3400.0]
        self.amps = [1.0, 0.5, 0.5, 0.5]
        self.on = [True, False, False, False]
        self.drive = (0.5, 0.5)          # relative position on the plate
        self.zeta = 0.02
        self.shake = 1.0
        self.width_cm = 30.0
        self.height_cm = 30.0
        self.thick_mm = 1.0
        self.material = "Steel"
        self.paused = False
        self.show_field = True           # read live by the worker
        self.cmap = COLORMAPS[0]
        self.show_sand = True
        self.quality = 1.0
        self.capture_cross = True
        self._render_size = (GRID, GRID) # computed image (nx, ny): plate pixels x quality
        self._plate_px = (GRID, GRID)    # plate size on screen in device pixels

        # Shared with the physics thread. _result and _sand are swapped
        # atomically as whole objects; parameter edits bump _version.
        self._lock = threading.Lock()
        self._version = 0
        self._reset_sand = False
        self._stop = threading.Event()
        self._result = None              # (id, spectrum)
        self._frame = None               # (frame id, RGB plate image)
        self._sand = np.random.rand(N_SAND, 2)
        self._shown = -1
        self._shown_frame = -1
        self._bg = self._spec_bg = None
        self._spec_dirty = True
        self._last_tick = time.perf_counter()
        self._export_thread = None
        self._export_status = ""

        self.spec_f = np.linspace(1.0, F_MAX, 3000)
        self._build_ui()
        self._refresh_geometry()
        self._worker = threading.Thread(target=self._physics_loop, daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- physics
    def _state(self):
        return dict(freqs=list(self.freqs), amps=list(self.amps), on=list(self.on),
                    drive=self.drive, zeta=self.zeta, shake=self.shake, material=self.material,
                    width_cm=self.width_cm, height_cm=self.height_cm, thick_mm=self.thick_mm)

    def _params(self):
        return params_of(self._state())

    def _physics_loop(self):
        rng = np.random.default_rng()
        sand = self._sand
        amp = np.zeros((GRID, GRID), np.float32)
        step_scale = np.ones(2)
        plate, plate_key, seen, spec_key = None, None, -1, None
        view, view_key, base_key = amp, None, None
        dt = 1.0 / SAND_RATE
        next_t = time.perf_counter()
        frame_id = 0
        while not self._stop.is_set():
            with self._lock:
                version = self._version
                p = self._params() if version != seen else None
                reset, self._reset_sand = self._reset_sand, False
            if reset:
                sand = rng.random((N_SAND, 2))
            if p is not None:
                if p["plate"] != plate_key:
                    plate_key = p["plate"]
                    plate = Plate(*plate_key)
                    lx, ly = plate_key[:2]
                    step_scale = np.sqrt(lx * ly) / np.array([lx, ly])   # isotropic in metres
                size = self._render_size
                view = plate.field(p["gens"], p["drive"], p["zeta"], *size)
                view_key = size
                key = (plate_key, p["drive"], p["zeta"])     # the spectrum ignores the generators
                if key != spec_key:
                    spec_key = key
                    self._result = (version, plate.spectrum(self.spec_f, p["drive"], p["zeta"]))
                seen, gens = version, p["gens"]
            size = self._render_size
            if plate is not None and view_key != size:
                view = plate.field(gens, spec_key[1], spec_key[2], *size)
                view_key = size
            cmap = self.cmap if self.show_field else None
            if base_key != (id(view), cmap, self._plate_px):
                base_key = (id(view), cmap, self._plate_px)
                amp = upscale(view, *self._plate_px)    # the sand reads this too: one field per change
                base = field_rgba(amp, cmap)
            if not self.paused:
                sand = step_sand(sand, amp, self.shake * step_scale, rng)
                self._sand = sand
            frame_id += 1
            self._frame = (frame_id, compose(base, sand, self.show_sand))

            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.perf_counter()    # fell behind; don't try to catch up

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        mpl.style.use("dark_background")
        mpl.rcParams.update({"axes.facecolor": PANEL, "figure.facecolor": BG,
                             "axes.edgecolor": MUTED, "text.color": FG,
                             "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG})
        self.qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        apply_dark_theme(self.qapp)
        self.qapp.aboutToQuit.connect(self._shutdown)

        # ---- plots (matplotlib only draws the plate and the spectrum)
        self.fig = fig = Figure(figsize=(8, 8.5))
        canvas = FigureCanvasQTAgg(fig)
        self.ax_plate = ax = fig.add_axes([0.1, 0.30, 0.86, 0.63])
        ax.set_aspect("equal", adjustable="box")   # 1 cm is the same on both axes
        ax.set_facecolor("#0b0b0b")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        self._rgb = None                 # latest plate image, pasted straight into the canvas
        (self.drive_mark,) = ax.plot([], [], "+", color="#00e5ff", ms=14, mew=2, animated=True)
        fig.text(0.1, 0.965, "Click the plate to move the exciter", color=MUTED, fontsize=9)

        self.ax_spec = fig.add_axes([0.1, 0.06, 0.86, 0.15])
        (self.spec_line,) = self.ax_spec.semilogy(self.spec_f, np.ones_like(self.spec_f),
                                                  "-", color="#9fa8da", lw=0.8, animated=True)
        self.ax_spec.set_xlim(0, F_MAX)
        self.ax_spec.set_xlabel("Frequency [Hz]")
        self.ax_spec.set_yticks([])
        self.ax_spec.minorticks_off()
        self.gen_lines = [self.ax_spec.axvline(f, color=c, lw=1.5, animated=True)
                          for f, c in zip(self.freqs, GEN_COLORS)]

        # ---- control panel (native Qt widgets)
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)

        def group(title):
            box = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(box)
            grid.setColumnStretch(1, 1)
            grid.setColumnMinimumWidth(0, 105)     # line up sliders across groups
            vbox.addWidget(box)
            return grid

        grid = group("Generators")
        self.gen_checks, self.f_controls, self.a_controls = [], [], []
        for i, c in enumerate(GEN_COLORS):
            check = QtWidgets.QCheckBox(f"f{i + 1} [Hz]")
            check.setChecked(self.on[i])
            check.setStyleSheet(f"color: {c};")
            check.toggled.connect(lambda on, i=i: self._set_item("on", i, on))
            self.gen_checks.append(check)
            self.f_controls.append(ValueControl(
                grid, 2 * i, check, 20, F_MAX, self.freqs[i], 1, 2, c,
                lambda v, i=i: self._set_item("freqs", i, v)))
            self.a_controls.append(ValueControl(
                grid, 2 * i + 1, f"A{i + 1}", 0, 1, self.amps[i], 0.01, 3, c,
                lambda v, i=i: self._set_item("amps", i, v)))

        grid = group("Plate")
        self.c_width = ValueControl(grid, 0, "Width [cm]", 5, 80, self.width_cm, 0.5, 2, GREY,
                                    lambda v: self._set_plate("width_cm", v))
        self.c_height = ValueControl(grid, 1, "Height [cm]", 5, 80, self.height_cm, 0.5, 2, GREY,
                                     lambda v: self._set_plate("height_cm", v))
        self.c_thick = ValueControl(grid, 2, "Thickness [mm]", 0.2, 5.0, self.thick_mm, 0.1, 3, GREY,
                                    lambda v: self._set_plate("thick_mm", v))
        self.c_zeta = ValueControl(grid, 3, "Damping", 0.002, 0.2, self.zeta, 0.001, 4, GREY,
                                   lambda v: self._set_attr("zeta", v))
        self.cmb_material = QtWidgets.QComboBox()
        self.cmb_material.addItems(list(MATERIALS))
        self.cmb_material.setCurrentText(self.material)
        self.cmb_material.currentTextChanged.connect(lambda m: self._set_plate("material", m))
        grid.addWidget(QtWidgets.QLabel("Material"), 4, 0)
        grid.addWidget(self.cmb_material, 4, 1, 1, 2)

        # Exciter position in cm; ranges follow the plate size (see _refresh_geometry)
        grid = group("Exciter")
        self.c_drive_x = ValueControl(grid, 0, "x [cm]", 0, self.width_cm, self.drive[0] * self.width_cm,
                                      0.1, 2, CYAN, lambda v: self._set_drive(x=v / self.width_cm))
        self.c_drive_y = ValueControl(grid, 1, "y [cm]", 0, self.height_cm, self.drive[1] * self.height_cm,
                                      0.1, 2, CYAN, lambda v: self._set_drive(y=v / self.height_cm))

        grid = group("Sand")
        self.c_shake = ValueControl(grid, 0, "Shake", 0.1, 3.0, self.shake, 0.05, 2, GREY,
                                    lambda v: setattr(self, "shake", v))   # read live by the worker
        row = QtWidgets.QHBoxLayout()
        for name, attr in (("Field", "show_field"), ("Sand", "show_sand")):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(lambda on, attr=attr: setattr(self, attr, on))
            row.addWidget(check)
        row.addStretch()
        btn_reset = QtWidgets.QPushButton("Reset sand")
        btn_reset.clicked.connect(lambda: self._set_attr("_reset_sand", True, recompute=False))
        row.addWidget(btn_reset)
        grid.addLayout(row, 1, 0, 1, 3)
        row = QtWidgets.QHBoxLayout()
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(COLORMAPS)
        self.cmb_cmap.currentTextChanged.connect(lambda name: setattr(self, "cmap", name))
        row.addWidget(QtWidgets.QLabel("Colormap"))
        row.addWidget(self.cmb_cmap)
        row.addStretch()
        grid.addLayout(row, 4, 0, 1, 3)
        row = QtWidgets.QHBoxLayout()
        self.cmb_quality = QtWidgets.QComboBox()
        self.cmb_quality.addItems(list(QUALITY))
        self.cmb_quality.currentTextChanged.connect(self._set_quality)
        row.addWidget(QtWidgets.QLabel("Quality"))
        row.addWidget(self.cmb_quality)
        row.addStretch()
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._set_paused)
        btn_shot = QtWidgets.QPushButton("Screenshot")
        btn_shot.clicked.connect(self._screenshot)
        row.addWidget(self.btn_pause)
        row.addWidget(btn_shot)
        grid.addLayout(row, 2, 0, 1, 3)
        check = QtWidgets.QCheckBox("Cross in captures")
        check.setChecked(True)
        check.setToolTip("Draw the exciter cross in screenshots and videos")
        check.toggled.connect(lambda on: setattr(self, "capture_cross", on))
        grid.addWidget(check, 3, 0, 1, 3)

        grid = group("Video")
        self.spin_duration = QtWidgets.QDoubleSpinBox()
        self.spin_duration.setRange(0.5, 600)
        self.spin_duration.setDecimals(1)
        self.spin_duration.setValue(10)
        self.spin_duration.setSuffix(" s")
        btn_render = QtWidgets.QPushButton("Render video")
        btn_render.clicked.connect(self._start_export)
        self.export_label = QtWidgets.QLabel("")
        self.export_label.setStyleSheet(f"color: {MUTED_TEXT};")
        self.export_label.setWordWrap(True)
        grid.addWidget(QtWidgets.QLabel("Length"), 0, 0)
        grid.addWidget(self.spin_duration, 0, 1)
        grid.addWidget(btn_render, 0, 2)
        grid.addWidget(self.export_label, 1, 0, 1, 3)
        vbox.addStretch()
        self.controls = [*self.f_controls, *self.a_controls, self.c_width, self.c_height,
                         self.c_thick, self.c_zeta, self.c_drive_x, self.c_drive_y, self.c_shake]
        for i in range(len(GEN_COLORS)):
            self.f_controls[i].key = ("freqs", i)
            self.a_controls[i].key = ("amps", i)
        for c, key in ((self.c_width, ("width_cm", None)), (self.c_height, ("height_cm", None)),
                       (self.c_thick, ("thick_mm", None)), (self.c_zeta, ("zeta", None)),
                       (self.c_drive_x, ("drive_cm", 0)), (self.c_drive_y, ("drive_cm", 1)),
                       (self.c_shake, ("shake", None))):
            c.key = key

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(panel.sizeHint().width() + 20)

        # ---- window
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Chladni Patterns")
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(NavigationToolbar2QT(canvas, left))
        lv.addWidget(canvas, 1)
        self.canvas = canvas
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        self.window.setCentralWidget(splitter)
        self.window.resize(1400, 860)

        canvas.mpl_connect("scroll_event", scroll_handler)     # Ctrl+scroll zooms the plots
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("draw_event", self._on_draw)
        self.timer = canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _shutdown(self):
        """Stop background work before Python finalises (avoids crashes at exit)."""
        self._stop.set()
        self._worker.join(2)
        if self._export_thread is not None:
            self._export_thread.join(2)
        POOL.shutdown(wait=True, cancel_futures=True)

    def show(self):
        self.window.show()
        self.qapp.exec()

    def _set_attr(self, name, value, recompute=True):
        with self._lock:
            setattr(self, name, value)
            self._version += recompute

    def _set_item(self, name, i, value):
        with self._lock:
            getattr(self, name)[i] = value
            self._version += 1
        self._refresh_gen_lines()

    def _set_plate(self, name, value):
        self._set_attr(name, value)
        self._refresh_geometry()

    def _set_drive(self, x=None, y=None):
        dx, dy = self.drive
        drive = (float(np.clip(dx if x is None else x, 0, 1)),
                 float(np.clip(dy if y is None else y, 0, 1)))
        if drive != self.drive:
            self._set_attr("drive", drive)
            self._refresh_drive_mark()

    def _set_quality(self, name):
        self.quality = QUALITY[name]
        self._update_render_size()
        self.fig.canvas.draw_idle()

    def _update_render_size(self):
        """Match the display image to the plate's on-screen pixels times the quality."""
        _, _, w, h = plate_rect(self.ax_plate)
        self._plate_px = (w, h)
        self._render_size = (max(8, round(w * self.quality)), max(8, round(h * self.quality)))

    def _set_paused(self, paused):
        self.paused = paused
        self.btn_pause.setText("Resume" if paused else "Pause")

    def _screenshot(self):
        VIDEO_DIR.mkdir(exist_ok=True)
        path = VIDEO_DIR / f"chladni_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        if not self.capture_cross:
            self.drive_mark.set_visible(False)
            self._blit_plate()
        ok = self.canvas.grab().save(str(path))     # grabs the blitted frame as shown
        if not self.capture_cross:
            self.drive_mark.set_visible(True)
            self._blit_plate()
        self._export_status = f"Saved {path.relative_to(VIDEO_DIR.parent)}" if ok else "Screenshot failed"

    def _on_click(self, event):
        # Go through the controls so they stay in sync; they call _set_drive
        if event.inaxes is self.ax_plate and event.xdata is not None and event.button == 1:
            self.c_drive_x.set_value(event.xdata)
            self.c_drive_y.set_value(event.ydata)

    def _refresh_gen_lines(self):
        for line, f, on in zip(self.gen_lines, self.freqs, self.on):
            line.set_xdata([f, f])
            line.set_alpha(1.0 if on else 0.2)
        self._spec_dirty = True

    def _refresh_drive_mark(self):
        self.drive_mark.set_data([self.drive[0] * self.width_cm], [self.drive[1] * self.height_cm])

    def _refresh_geometry(self):
        w, h = self.width_cm, self.height_cm
        self.ax_plate.set_xlim(0, w)
        self.ax_plate.set_ylim(0, h)
        # Rescale the exciter controls, keeping the exciter at the same relative spot
        self.c_drive_x.set_range(0, w)
        self.c_drive_y.set_range(0, h)
        self.c_drive_x.set_value(self.drive[0] * w, notify=False)
        self.c_drive_y.set_value(self.drive[1] * h, notify=False)
        f10 = np.pi / 2 * plate_speed(self.thick_mm / 1000, self.material) / (max(w, h) / 100) ** 2
        self.ax_plate.set_title(f"{w:g} × {h:g} cm, {self.thick_mm:g} mm {self.material}"
                                f"   ·   lowest mode {f10:.1f} Hz")
        self._refresh_drive_mark()
        self._refresh_gen_lines()
        self.fig.canvas.draw_idle()                 # title and limits are static artists

    # ----------------------------------------------------------- video export
    def _start_export(self):
        if self._export_thread is not None and self._export_thread.is_alive():
            return
        self.spin_duration.interpretText()        # take a typed value even without Enter
        duration = self.spin_duration.value()
        self.btn_pause.setChecked(False)          # recording runs the live sim too
        lfos = [spec for c in self.controls if (spec := c.lfo_spec())]
        for c in self.controls:                   # retrigger: live LFOs restart with the video
            c.phase = 0.0
        with self._lock:
            state = apply_lfos(self._state(), lfos, 0.0)   # LFO'd values back at their centres
        size_cm = (state["width_cm"], state["height_cm"])
        args = (state, lfos, self._sand, size_cm, self.ax_plate.get_title(),
                self.cmap if self.show_field else None, self.show_sand, self.capture_cross, duration)
        self._export_thread = threading.Thread(target=self._export, args=args, daemon=True)
        self._export_thread.start()

    def _export(self, state, lfos, sand, size_cm, title, cmap, show_sand, show_cross, duration):
        """Re-run the simulation offline on a private Agg figure and write a video.

        LFOs restart at phase 0. Modulated plate size changes the physics only;
        the video frame keeps the plate's centre dimensions.
        """
        try:
            params = params_of(state)
            plate = Plate(*params["plate"])
            rng = np.random.default_rng()
            steps_per_frame = SAND_RATE / VIDEO_FPS
            n_frames = max(1, round(duration * VIDEO_FPS))

            # Plate fits a 1000 x 640 box with its true aspect ratio; fixed pixel
            # margins around it. Even frame dimensions keep H.264 happy.
            dpi = 100
            ratio = size_cm[0] / size_cm[1]
            plate_w, plate_h = (round(640 * ratio), 640) if ratio <= 1000 / 640 else (1000, round(1000 / ratio))
            left, right, bottom, top = 80, 30, 60, 40
            px_w, px_h = plate_w + left + right, plate_h + bottom + top
            right += px_w % 2
            top += px_h % 2
            px_w, px_h = px_w + px_w % 2, px_h + px_h % 2
            fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor=BG)
            FigureCanvasAgg(fig)
            ax = fig.add_axes([left / px_w, bottom / px_h, plate_w / px_w, plate_h / px_h])
            ax.set_facecolor("#0b0b0b")
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.tick_params(colors=FG)
            ax.set_xlabel("x [cm]", color=FG)
            ax.set_ylabel("y [cm]", color=FG)
            ax.set_title(title, color=FG, fontsize=10)
            fig.canvas.draw()                    # lay out to learn the axes' pixel size
            rect = plate_rect(ax)
            size = rect[2:]
            amp = plate.field(params["gens"], params["drive"], params["zeta"], *size)
            base = field_rgba(amp, cmap)
            (mark,) = ax.plot(params["drive"][0] * size_cm[0], params["drive"][1] * size_cm[1],
                              "+", color="#00e5ff", ms=14, mew=2, animated=True)
            mark.set_visible(show_cross)
            ax.set_xlim(0, size_cm[0])
            ax.set_ylim(0, size_cm[1])
            clock = ax.text(0.99, 0.01, "", transform=ax.transAxes, ha="right", va="bottom",
                            color=FG, fontsize=9, animated=True,
                            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.8, pad=2))
            fig.canvas.draw()                    # static parts once, then blit per frame
            bg = fig.canvas.copy_from_bbox(fig.bbox)

            VIDEO_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                path = VIDEO_DIR / f"chladni_{stamp}.mp4"
                proc = subprocess.Popen(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                     "-s", f"{px_w}x{px_h}", "-r", str(VIDEO_FPS), "-i", "-",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                    stdin=subprocess.PIPE)
            else:                                # no ffmpeg: fall back to a GIF via Pillow
                from PIL import Image
                path = VIDEO_DIR / f"chladni_{stamp}.gif"
                proc, gif_frames = None, []

            carry = 0.0
            try:
                for frame in range(n_frames):
                    if self._stop.is_set():
                        return
                    st = apply_lfos(state, lfos, frame / VIDEO_FPS)
                    p = params_of(st)
                    if p != params:
                        if p["plate"] != params["plate"]:
                            plate = Plate(*p["plate"])
                        amp = plate.field(p["gens"], p["drive"], p["zeta"], *size)
                        base = field_rgba(amp, cmap)
                        mark.set_data([p["drive"][0] * size_cm[0]], [p["drive"][1] * size_cm[1]])
                        params = p
                    lx, ly = p["plate"][:2]
                    step_scale = st["shake"] * np.sqrt(lx * ly) / np.array([lx, ly])
                    carry += steps_per_frame
                    while carry >= 1.0:
                        sand = step_sand(sand, amp, step_scale, rng)
                        carry -= 1.0
                    clock.set_text(f"t = {frame / VIDEO_FPS:5.2f} s")
                    fig.canvas.restore_region(bg)
                    paste(fig.canvas, rect, compose(base, sand, show_sand))
                    for artist in (*ax.spines.values(), mark, clock):
                        ax.draw_artist(artist)
                    rgba = fig.canvas.buffer_rgba()
                    if proc:
                        proc.stdin.write(rgba)
                    else:
                        gif_frames.append(Image.fromarray(np.asarray(rgba)[..., :3]).quantize())
                    self._export_status = f"Rendering… {100 * (frame + 1) // n_frames}%"
            finally:
                if proc:
                    proc.stdin.close()
                    proc.wait()
            if proc and proc.returncode:
                raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
            if not proc:
                gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                                   duration=round(1000 / VIDEO_FPS), loop=0)
            self._export_status = f"Saved {path.relative_to(VIDEO_DIR.parent)}"
        except Exception as exc:                 # report instead of dying silently
            self._export_status = f"Export failed: {exc}"

    # -------------------------------------------------------------- rendering
    # Plate artists are animated and blitted on top of a cached background, so
    # each frame redraws only the plate instead of the whole figure.
    def _draw_plate_artists(self):
        if self._rgb is not None:
            paste(self.fig.canvas, plate_rect(self.ax_plate), self._rgb)
        for artist in (*self.ax_plate.spines.values(), self.drive_mark):
            self.ax_plate.draw_artist(artist)

    def _draw_spec_artists(self):
        for artist in (self.spec_line, *self.gen_lines):
            self.ax_spec.draw_artist(artist)

    def _on_draw(self, _event):
        canvas = self.fig.canvas
        self._bg = canvas.copy_from_bbox(self.ax_plate.bbox)
        self._spec_bg = canvas.copy_from_bbox(self.ax_spec.bbox)
        self._update_render_size()
        self._draw_plate_artists()
        self._draw_spec_artists()

    def _blit_plate(self):
        canvas = self.fig.canvas
        canvas.restore_region(self._bg)
        self._draw_plate_artists()
        canvas.blit(self.ax_plate.bbox)

    def _tick(self):
        now = time.perf_counter()
        dt, self._last_tick = min(now - self._last_tick, 0.1), now
        for control in self.controls:
            control.lfo_step(dt)

        res = self._result
        if res is not None and res[0] != self._shown:
            self._shown, spec = res
            self.spec_line.set_ydata(spec)
            lo, hi = spec.min(), spec.max()            # no y ticks: limits need no full redraw
            self.ax_spec.set_ylim(lo / 2, hi * 2)
            self._spec_dirty = True
        if self.export_label.text() != self._export_status:
            self.export_label.setText(self._export_status)

        canvas = self.fig.canvas
        if self._bg is None or not canvas.supports_blit:
            canvas.draw_idle()
            return
        if self._spec_dirty:
            self._spec_dirty = False
            canvas.restore_region(self._spec_bg)
            self._draw_spec_artists()
            canvas.blit(self.ax_spec.bbox)
        frame = self._frame
        if frame is not None and frame[0] != self._shown_frame:
            self._shown_frame, self._rgb = frame
            self._blit_plate()


if __name__ == "__main__":
    Chladni().show()
