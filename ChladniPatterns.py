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

Physics (field, spectrum, sand) runs in a background thread; the GUI thread
only blits the latest results. Controls are native Qt widgets beside the plot.

Controls
    sliders        frequency / amplitude per generator, plate size / thickness /
                   damping, exciter x / y position, shake strength
    number boxes   type an exact value (applied on Enter or when leaving the box);
                   arrow keys / mouse wheel step it
    check boxes    enable generators (the coloured f labels), toggle field / sand
    Material       plate material
    click plate    move the exciter (updates the x / y controls; breaks symmetry)
    Reset sand     scatter the sand uniformly again
    Render video   simulate the current setup for the chosen length (starting
                   from the current sand) and save it to renders/
"""
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.style
from matplotlib.backend_bases import scroll_handler
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

GRID = 180          # plate resolution (pixels per side)
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

x = np.linspace(0.0, 1.0, GRID)


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
        self.bx = np.cos(np.pi * self.kx[:, None] * x[None, :])    # (nx, GRID)
        self.by = np.cos(np.pi * self.ky[:, None] * x[None, :])    # (ny, GRID)
        self.f = self.fy * self.ky[:, None] ** 2 + self.fx * self.kx[None, :] ** 2   # [n_y, m_x]
        self.active = self.f > 0                     # drop the rigid (0,0) mode
        self.audible = self.active & (self.f < 2.0 * F_MAX)

    def drive_weights(self, drive):
        dx, dy = drive
        return np.outer(np.cos(np.pi * self.ky * dy), np.cos(np.pi * self.kx * dx))

    def transfer(self, f, zeta):
        """Damped-oscillator response of every mode to a sine at frequency f."""
        h = np.zeros_like(self.f, dtype=complex)
        fm = self.f[self.active]
        h[self.active] = 1.0 / (fm ** 2 - f ** 2 + 2j * zeta * fm * f)
        return h

    def field(self, gens, drive, zeta):
        """Normalised vibration amplitude on the GRID x GRID plate."""
        phi_d = self.drive_weights(drive)
        energy = np.zeros((GRID, GRID))
        for f, a, on in gens:
            if on and a > 0:
                coeffs = a * phi_d * self.transfer(f, zeta)
                u = self.by.T @ coeffs @ self.bx             # complex displacement
                energy += np.abs(u) ** 2
        amp = np.sqrt(energy)
        peak = amp.max()
        return amp / peak if peak > 0 else amp

    def spectrum(self, f, drive, zeta):
        """Drive-point response vs frequency, to show where the resonances are."""
        w = self.drive_weights(drive)[self.audible] ** 2
        fm = self.f[self.audible]
        h = 1.0 / (fm[None, :] ** 2 - f[:, None] ** 2 + 2j * zeta * fm[None, :] * f[:, None])
        return (w[None, :] * np.abs(h) ** 2).sum(axis=1) + 1e-30


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

    def _from_slider(self, pos):
        with QtCore.QSignalBlocker(self.spin):
            self.spin.setValue(self.lo + pos * self.step)
        self.on_change(self.spin.value())

    def _from_spin(self, value):
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(round((value - self.lo) / self.step))
        self.on_change(value)


def step_sand(sand, amp, step_scale, rng):
    """One random-walk step; returns a new array (the GUI may hold the old one)."""
    idx = np.clip(np.rint(sand * (GRID - 1)).astype(int), 0, GRID - 1)
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

        # Shared with the physics thread. _result and _sand are swapped
        # atomically as whole objects; parameter edits bump _version.
        self._lock = threading.Lock()
        self._version = 0
        self._reset_sand = False
        self._stop = threading.Event()
        self._result = None              # (version, amp, spectrum)
        self._sand = np.random.rand(N_SAND, 2)
        self._shown = -1
        self._bg = None
        self._export_thread = None
        self._export_status = ""

        self.spec_f = np.linspace(1.0, F_MAX, 3000)
        self._build_ui()
        self._refresh_geometry()
        self._worker = threading.Thread(target=self._physics_loop, daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- physics
    def _params(self):
        return dict(
            gens=tuple(zip(self.freqs, self.amps, self.on)),
            drive=self.drive,
            zeta=self.zeta,
            plate=(self.width_cm / 100, self.height_cm / 100, self.thick_mm / 1000, self.material),
        )

    def _physics_loop(self):
        rng = np.random.default_rng()
        sand = self._sand
        amp = np.zeros((GRID, GRID))
        step_scale = np.ones(2)
        plate, plate_key, seen = None, None, -1
        dt = 1.0 / SAND_RATE
        next_t = time.perf_counter()
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
                amp = plate.field(p["gens"], p["drive"], p["zeta"])
                spec = plate.spectrum(self.spec_f, p["drive"], p["zeta"])
                self._result = (version, amp, spec)
                seen = version
            sand = step_sand(sand, amp, self.shake * step_scale, rng)
            self._sand = sand

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
        self.qapp.aboutToQuit.connect(self._stop.set)

        # ---- plots (matplotlib only draws the plate and the spectrum)
        self.fig = fig = Figure(figsize=(8, 8.5))
        canvas = FigureCanvasQTAgg(fig)
        self.ax_plate = ax = fig.add_axes([0.1, 0.30, 0.86, 0.63])
        ax.set_facecolor("#0b0b0b")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        self.img = ax.imshow(np.zeros((GRID, GRID)), origin="lower", cmap="inferno",
                             vmin=0, vmax=1, interpolation="bilinear", animated=True)
        self.scat = ax.scatter([], [], s=0.4, c="#f3e3c0", lw=0, animated=True)
        (self.drive_mark,) = ax.plot([], [], "+", color="#00e5ff", ms=14, mew=2, animated=True)
        fig.text(0.1, 0.965, "Click the plate to move the exciter", color=MUTED, fontsize=9)

        self.ax_spec = fig.add_axes([0.1, 0.06, 0.86, 0.15])
        (self.spec_line,) = self.ax_spec.semilogy(self.spec_f, np.ones_like(self.spec_f),
                                                  "-", color="#9fa8da", lw=0.8)
        self.ax_spec.set_xlim(0, F_MAX)
        self.ax_spec.set_xlabel("Frequency [Hz]")
        self.ax_spec.set_yticks([])
        self.ax_spec.minorticks_off()
        self.gen_lines = [self.ax_spec.axvline(f, color=c, lw=1.5)
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
        for name, artist in (("Field", self.img), ("Sand", self.scat)):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(artist.set_visible)
            row.addWidget(check)
        row.addStretch()
        btn_reset = QtWidgets.QPushButton("Reset sand")
        btn_reset.clicked.connect(lambda: self._set_attr("_reset_sand", True, recompute=False))
        row.addWidget(btn_reset)
        grid.addLayout(row, 1, 0, 1, 3)

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

    def _on_click(self, event):
        # Go through the controls so they stay in sync; they call _set_drive
        if event.inaxes is self.ax_plate and event.xdata is not None and event.button == 1:
            self.c_drive_x.set_value(event.xdata)
            self.c_drive_y.set_value(event.ydata)

    def _refresh_gen_lines(self):
        for line, f, on in zip(self.gen_lines, self.freqs, self.on):
            line.set_xdata([f, f])
            line.set_alpha(1.0 if on else 0.2)
        self.fig.canvas.draw_idle()

    def _refresh_drive_mark(self):
        self.drive_mark.set_data([self.drive[0] * self.width_cm], [self.drive[1] * self.height_cm])

    def _refresh_geometry(self):
        w, h = self.width_cm, self.height_cm
        self.img.set_extent((0, w, 0, h))
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

    # ----------------------------------------------------------- video export
    def _start_export(self):
        if self._export_thread is not None and self._export_thread.is_alive():
            return
        self.spin_duration.interpretText()        # take a typed value even without Enter
        duration = self.spin_duration.value()
        with self._lock:
            params = self._params()
        size_cm = (self.width_cm, self.height_cm)
        args = (params, self._sand, self.shake, size_cm, self.ax_plate.get_title(),
                self.img.get_visible(), self.scat.get_visible(), duration)
        self._export_thread = threading.Thread(target=self._export, args=args, daemon=True)
        self._export_thread.start()

    def _export(self, params, sand, shake, size_cm, title, show_field, show_sand, duration):
        """Re-run the simulation offline on a private Agg figure and write a video."""
        try:
            lx, ly = params["plate"][:2]
            plate = Plate(*params["plate"])
            amp = plate.field(params["gens"], params["drive"], params["zeta"])
            step_scale = shake * np.sqrt(lx * ly) / np.array([lx, ly])
            rng = np.random.default_rng()
            steps_per_frame = SAND_RATE / VIDEO_FPS
            n_frames = max(1, round(duration * VIDEO_FPS))

            # Even pixel dimensions keep H.264 happy
            dpi = 100
            px_h = 720
            px_w = int(round(px_h * min(max(size_cm[0] / size_cm[1], 0.5), 2.5) * 0.9 + 140))
            px_w += px_w % 2
            fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor=BG)
            FigureCanvasAgg(fig)
            ax = fig.add_axes([0.1, 0.08, 0.86, 0.84])
            ax.set_facecolor("#0b0b0b")
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.tick_params(colors=FG)
            ax.set_xlabel("x [cm]", color=FG)
            ax.set_ylabel("y [cm]", color=FG)
            ax.set_title(title, color=FG, fontsize=10)
            img = ax.imshow(amp, origin="lower", cmap="inferno", vmin=0, vmax=1,
                            interpolation="bilinear", extent=(0, size_cm[0], 0, size_cm[1]))
            img.set_visible(show_field)
            scat = ax.scatter(sand[:, 0] * size_cm[0], sand[:, 1] * size_cm[1],
                              s=0.6, c="#f3e3c0", lw=0, animated=True)
            scat.set_visible(show_sand)
            ax.plot(params["drive"][0] * size_cm[0], params["drive"][1] * size_cm[1],
                    "+", color="#00e5ff", ms=14, mew=2)
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
                    carry += steps_per_frame
                    while carry >= 1.0:
                        sand = step_sand(sand, amp, step_scale, rng)
                        carry -= 1.0
                    scat.set_offsets(sand * size_cm)
                    clock.set_text(f"t = {frame / VIDEO_FPS:5.2f} s")
                    fig.canvas.restore_region(bg)
                    ax.draw_artist(scat)
                    ax.draw_artist(clock)
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
        for artist in (self.img, self.scat, self.drive_mark):
            self.ax_plate.draw_artist(artist)

    def _on_draw(self, _event):
        self._bg = self.fig.canvas.copy_from_bbox(self.ax_plate.bbox)
        self._draw_plate_artists()

    def _tick(self):
        res = self._result
        if res is not None and res[0] != self._shown:
            self._shown, amp, spec = res
            self.img.set_data(amp)
            self.spec_line.set_ydata(spec)
            self.ax_spec.relim()
            self.ax_spec.autoscale_view(scalex=False)
            self.fig.canvas.draw_idle()
        if self.export_label.text() != self._export_status:
            self.export_label.setText(self._export_status)
        if self.scat.get_visible():
            self.scat.set_offsets(self._sand * (self.width_cm, self.height_cm))

        canvas = self.fig.canvas
        if self._bg is None or not canvas.supports_blit:
            canvas.draw_idle()
            return
        canvas.restore_region(self._bg)
        self._draw_plate_artists()
        canvas.blit(self.ax_plate.bbox)


if __name__ == "__main__":
    Chladni().show()
