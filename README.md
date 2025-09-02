
# Quarter-Car RLDA Predictor (Starter Kit)

This starter kit simulates **quarter-car vertical dynamics** to generate **Road Load Data (RLDA)** from suspension parameters and road inputs. It outputs time histories, PSDs, and fatigue-oriented summaries (rainflow + Equivalent Damage Load).

## Contents
- `src/rlda_quarter_car.py` — core simulator and metrics
- `run_quarter_car.py` — simple CLI wrapper
- `templates/suspension_params.csv` — parameter template for your corner
- `templates/damper_map_example.csv` — optional damper dyno (velocity [m/s] vs force [N])
- `templates/road_profile.csv` — example road profile format (t [s], z_r [m])
- `examples/run_config.json` — example run configuration

## Quick start
1. Edit `templates/suspension_params.csv` with your values (or place your own CSV elsewhere)
2. (Optional) Provide `damper_map.csv` (two columns: `v_mps`, `F_N`) for asymmetric/nonlinear damping
3. (Optional) Provide measured `road_profile.csv` (two columns: `t`, `z_r`); else use ISO class generator
4. Run from terminal:

```bash
python run_quarter_car.py   --params templates/suspension_params.csv   --duration 20   --fs 1000   --speed 20   --iso_class C   --seed 42   --out out/quarter_car_C_20ms
```

Or, if you have a measured road profile:
```bash
python run_quarter_car.py   --params templates/suspension_params.csv   --road templates/road_profile.csv   --fs 1000   --out out/quarter_car_measured
```

## Outputs (in the chosen `--out` folder)
- `time_history.csv` — t, z_s, z_u, dz_s, dz_u, z_r, Fz
- `psd.csv` — frequency [Hz], PSD of Fz [N^2/Hz] (Welch)
- `rainflow.csv` — ranges & counts for Fz
- `metrics.json` — summary (RMS, peak, kurtosis, EDL)

## EDL (Equivalent Damage Load)
We compute EDL for Fz using a Wöhler exponent `m` (default 6 for steel-like) and a reference cycle count `N_ref` (default 1e6). Configure via CLI.

## Notes
- The quarter-car includes: spring, tire stiffness, bi-linear damper (or map), bump-stop (gap + rate), and optional tire nonlinearity.
- Road generator approximates ISO 8608 classes via filtered colored noise with calibrated RMS; for certification, replace with measured profiles.
- Extend to half-car for roll/ARB effects or to MBD for full wheel 6-DOF loads.

© 2025-09-02
