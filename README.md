# Chladni Patterns

An interactive Chladni plate simulator. A rectangular plate with real dimensions and a real material is driven at one point by up to four sine generators. Sand on the plate drifts away from the vibrating regions and collects on the nodal lines, as it does on a real plate.

Mode frequencies follow Kirchhoff plate theory, with free-edge-like cosine mode shapes:

```
f_mn = (π/2) · sqrt(D / (ρh)) · ((m/Lx)² + (n/Ly)²),   D = E h³ / (12 (1 − ν²))
```

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`, `PySide6-Essentials` (see `requirements.txt`)
- `ffmpeg` on PATH, for MP4 export. Without it, videos are saved as GIFs, which needs Pillow.

## Run

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python ChladniPatterns.py
```

## Controls

| Control | What it does |
|---|---|
| **Generators** f1–f4 / A1–A4 | Frequency and amplitude of each sine generator. The coloured check box turns a generator on or off. |
| **Plate** | Width, height, thickness, damping and material (steel, aluminium, brass, glass). |
| **Exciter** x / y | Where the plate is driven. You can also click the plate to move it there. |
| **Sand** Shake | How hard the grains are kicked. |
| Field / Sand | Show or hide the vibration field or the sand. |
| Reset sand | Scatter the sand evenly again. |
| Colormap | Colour scheme of the field: inferno, magma, plasma, viridis, cividis, turbo, twilight, coolwarm, ocean, bone or gray. Screenshots and videos use it too. |
| Quality | Resolution the vibration field is computed at: Full, 1/2 or 1/4 of the plate's size on screen. Lower settings are smoothly scaled up, and sand is always drawn at full resolution. Lower it if sliders or LFOs feel slow. |
| Pause / Resume | Freeze the sand. The field still updates when you change settings. |
| Screenshot | Save the plot area as a PNG to `renders/`. |
| Cross in captures | Choose whether screenshots and videos show the exciter cross. |
| **Video** Length + Render video | Simulate the current setup for the chosen length, starting from the current sand, and save an MP4 to `renders/`. |
| `~` (next to each slider) | LFO. Click it to turn it on or off. The arrow sets Rate (Hz) and Depth (± % of the slider's range). The slider sets the centre value the LFO swings around. |

Starting a render resumes a paused simulation and restarts every LFO at phase 0, both on screen and in the video. Plate-size LFOs change the physics in the video, but the video keeps the plate's base dimensions.

Number boxes take typed values when you press Enter or click away. Arrow keys and the mouse wheel step them. Ctrl+scroll zooms the plots.

The spectrum strip under the plate shows the plate's response at the exciter. Its peaks are the resonances, and the coloured lines mark the generator frequencies.

## How it works

- A background thread runs the physics. Each generator's field is computed in parallel in a thread pool and cached, the spectrum is recomputed only when it can change, and the sand is drawn straight into one RGB image.
- The GUI thread only advances the LFOs and blits the finished images, so the interface stays responsive.
- Video export re-runs the simulation on a separate off-screen figure and sends the frames to ffmpeg.
