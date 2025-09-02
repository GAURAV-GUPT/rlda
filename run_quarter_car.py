
import argparse
import os
import json
import pandas as pd
import numpy as np
from src.rlda_quarter_car import QuarterCarRLDA, QuarterCarParams, DamperModel

parser = argparse.ArgumentParser()
parser.add_argument('--params', required=True, help='CSV with suspension params')
parser.add_argument('--corner', default=None, help='corner_id to pick from CSV')
parser.add_argument('--road', default='', help='CSV with t,z_r; if empty, ISO generator is used')
parser.add_argument('--iso_class', default='C', help='ISO 8608 class A..H (if road not provided)')
parser.add_argument('--duration', type=float, default=20.0, help='duration [s]')
parser.add_argument('--fs', type=float, default=1000.0, help='sample rate [Hz]')
parser.add_argument('--speed', type=float, default=20.0, help='speed [m/s] (for metadata only)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', required=True, help='output folder')
parser.add_argument('--wohler_m', type=float, default=6.0)
parser.add_argument('--N_ref', type=float, default=1e6)
parser.add_argument('--use_damper_map', action='store_true')
parser.add_argument('--damper_map', default='', help='CSV with v_mps,F_N if using map')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

params = QuarterCarRLDA.load_params_csv(args.params, corner_id=args.corner)
map_df = None
if args.use_damper_map and args.damper_map:
    map_df = pd.read_csv(args.damper_map)

sim = QuarterCarRLDA(params, fs_hz=args.fs, duration_s=args.duration, seed=args.seed,
                     damper_model=DamperModel(params.c_bump, params.c_rebound, map_df))

if args.road:
    df_road = pd.read_csv(args.road)
    # resample road to fs and duration
    t = df_road.iloc[:,0].values
    zr = df_road.iloc[:,1].values
    t_uniform = np.arange(0, args.duration, 1.0/args.fs)
    zr_uniform = np.interp(t_uniform, t, zr)
else:
    zr_uniform = sim.road_from_iso(args.iso_class)

res = sim.simulate(zr_uniform)
metrics = QuarterCarRLDA.summarize(res['Fz'], fs=args.fs, m=args.wohler_m, N_ref=args.N_ref)

import pandas as pd
pd.DataFrame({
    't': res['t'], 'z_s': res['z_s'], 'z_u': res['z_u'], 'dz_s': res['dz_s'], 'dz_u': res['dz_u'],
    'z_r': res['z_r'], 'Fz': res['Fz']
}).to_csv(os.path.join(args.out, 'time_history.csv'), index=False)

# PSD
import numpy as np
f = np.array(metrics['psd']['f_Hz'])
P = np.array(metrics['psd']['P_N2_per_Hz'])
pd.DataFrame({'f_Hz': f, 'PSD_N2_per_Hz': P}).to_csv(os.path.join(args.out, 'psd.csv'), index=False)

# Rainflow ranges
pd.DataFrame({'range_N': metrics['rainflow_ranges_N']}).to_csv(os.path.join(args.out, 'rainflow.csv'), index=False)

# Metrics JSON
with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
    json.dump({k: v for k, v in metrics.items() if k not in ['psd','rainflow_ranges_N']}, f, indent=2)

# Metadata
with open(os.path.join(args.out, 'run_metadata.json'), 'w') as f:
    json.dump({
        'params_csv': args.params,
        'corner': args.corner,
        'road_csv': args.road,
        'iso_class': args.iso_class,
        'duration_s': args.duration,
        'fs_hz': args.fs,
        'speed_mps': args.speed,
        'seed': args.seed,
        'wohler_m': args.wohler_m,
        'N_ref': args.N_ref,
        'use_damper_map': args.use_damper_map,
        'damper_map': args.damper_map
    }, f, indent=2)

print('Done. Outputs at', args.out)
