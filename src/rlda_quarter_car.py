
import numpy as np
from dataclasses import dataclass
from scipy.signal import welch, butter, filtfilt
from scipy.integrate import solve_ivp
import pandas as pd

@dataclass
class QuarterCarParams:
    m_s: float
    m_u: float
    k_spring: float
    c_bump: float
    c_rebound: float
    k_tire: float
    bumpstop_gap: float = 0.0
    bumpstop_rate: float = 0.0
    tire_nl_knee: float = 0.0   # deflection at which tire stiffens
    tire_nl_gain: float = 0.0   # fractional stiffness gain beyond knee

class DamperModel:
    def __init__(self, c_bump, c_rebound, map_df=None):
        self.c_b = c_bump
        self.c_r = c_rebound
        self.map = map_df  # optional velocity-force map
        if self.map is not None:
            self.map = self.map.sort_values('v_mps')
            self.v = self.map['v_mps'].values
            self.F = self.map['F_N'].values

    def force(self, rel_vel):
        if self.map is not None:
            # interpolate force from map (asymmetric allowed)
            return np.interp(rel_vel, self.v, self.F)
        # simple bi-linear: bump (rel_vel>0), rebound (rel_vel<0)
        return (self.c_b if rel_vel >= 0 else self.c_r) * rel_vel

class QuarterCarRLDA:
    def __init__(self, params: QuarterCarParams, fs_hz=1000, duration_s=20.0, seed=42,
                 damper_model: DamperModel=None):
        self.p = params
        self.fs = fs_hz
        self.dt = 1.0/fs_hz
        self.N = int(duration_s * fs_hz)
        self.t = np.arange(self.N) * self.dt
        self.rng = np.random.default_rng(seed)
        self.damper = damper_model if damper_model is not None else DamperModel(params.c_bump, params.c_rebound)

    # ISO-like road generation; returns z_r
    def road_from_iso(self, iso_class='C'):
        # Roughness scaling per class (approximate): A..H
        classes = {'A':0.25,'B':0.5,'C':1.0,'D':2.0,'E':4.0,'G':8.0,'H':16.0}
        scale = classes.get(iso_class.upper(), 1.0)
        w = self.rng.standard_normal(self.N)
        # color to ~1/f^2 using double integration in frequency domain approximation by filtering
        # two cascaded low-pass filters to emphasize long wavelengths
        from scipy.signal import butter, sosfiltfilt
        sos1 = butter(2, 0.05, output='sos')
        sos2 = butter(2, 0.1, output='sos')
        z = sosfiltfilt(sos1, w)
        z = sosfiltfilt(sos2, z)
        z = z / np.std(z)
        z *= 0.01 * scale  # 10 mm std for class C baseline
        return z

    def simulate(self, z_r):
        p = self.p
        fs = self.fs
        t = self.t

        def tire_force(zu, zr):
            # base linear tire
            defl = zu - zr
            F = p.k_tire * defl
            if p.tire_nl_knee > 0 and abs(defl) > p.tire_nl_knee:
                gain = 1.0 + p.tire_nl_gain
                extra = (gain - 1.0) * p.k_tire * (abs(defl) - p.tire_nl_knee) * np.sign(defl)
                F += extra
            return F

        def bumpstop_force(zs, zu):
            jounce = (zs - zu)
            if p.bumpstop_gap <= 0:
                return 0.0
            pen = jounce - p.bumpstop_gap
            return p.bumpstop_rate * pen if pen > 0 else 0.0

        def ode(ti, y):
            zs, zu, dzs, dzu = y
            # interpolate zr
            i = min(int(ti*fs), len(z_r)-1)
            zr = z_r[i]
            rel = dzs - dzu
            Fd = self.damper.force(rel)
            Fs = p.k_spring * (zs - zu)
            Fb = bumpstop_force(zs, zu)
            Ft = tire_force(zu, zr)
            # EOM
            ddzs = (-Fd - Fs - Fb) / p.m_s
            ddzu = (Fd + Fs + Fb - Ft) / p.m_u
            return [dzs, dzu, ddzs, ddzu]

        y0 = [0.0, 0.0, 0.0, 0.0]
        sol = solve_ivp(ode, [0, self.t[-1]], y0, t_eval=self.t, max_step=1.0/self.fs, method='RK45')
        zs, zu, dzs, dzu = sol.y
        # Outputs
        Fz = p.k_tire * (zu - z_r)
        # add nonlinearity consistent with tire_force used in ODE for reporting
        if p.tire_nl_knee > 0:
            defl = zu - z_r
            mask = np.abs(defl) > p.tire_nl_knee
            Fz = Fz.astype(float)
            Fz[mask] += p.tire_nl_gain * p.k_tire * (np.abs(defl[mask]) - p.tire_nl_knee) * np.sign(defl[mask])
        Fz = Fz - np.mean(Fz)
        return {
            't': self.t,
            'z_s': zs,
            'z_u': zu,
            'dz_s': dzs,
            'dz_u': dzu,
            'z_r': z_r,
            'Fz': Fz
        }

    @staticmethod
    def psd(signal, fs, nperseg=2048):
        f, Pxx = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
        return f, Pxx

    @staticmethod
    def rainflow(series):
        # Simple rainflow range counter (ranges only)
        # Extract turning points
        x = np.asarray(series).flatten()
        tp = [x[0]]
        for i in range(1, len(x)-1):
            if (x[i]-x[i-1])*(x[i+1]-x[i]) < 0:
                tp.append(x[i])
        tp.append(x[-1])
        S = []
        counts = []
        for v in tp:
            S.append(v)
            while len(S) >= 3:
                X, Y, Z = S[-3], S[-2], S[-1]
                if abs(Y - X) <= abs(Z - Y):
                    rng = abs(Y - X)
                    counts.append(rng)
                    S.pop(-2)
                else:
                    break
        # collapse remaining
        for i in range(len(S)-1):
            counts.append(abs(S[i+1]-S[i]))
        return np.array(counts)

    @staticmethod
    def edl_from_ranges(ranges, m=6.0, N_ref=1e6):
        # Equivalent Damage Load with Wohler exponent m
        # Damage ~ sum( (S/2)^m ), EDL defined so that N_ref*(EDL/2)^m equals that sum
        if len(ranges) == 0:
            return 0.0
        damage = np.sum((ranges/2.0)**m)
        edl = 2.0 * (damage / N_ref) ** (1.0/m)
        return edl

    @staticmethod
    def summarize(Fz, fs, m=6.0, N_ref=1e6):
        import scipy.stats as st
        rms = float(np.sqrt(np.mean(Fz**2)))
        peak = float(np.max(np.abs(Fz)))
        kurt = float(st.kurtosis(Fz, fisher=False))
        f, P = QuarterCarRLDA.psd(Fz, fs)
        rf = QuarterCarRLDA.rainflow(Fz)
        edl = float(QuarterCarRLDA.edl_from_ranges(rf, m=m, N_ref=N_ref))
        return {
            'rms_N': rms,
            'peak_N': peak,
            'kurtosis': kurt,
            'edl_N': edl,
            'psd': {'f_Hz': f.tolist(), 'P_N2_per_Hz': P.tolist()},
            'rainflow_ranges_N': rf.tolist()
        }

    @staticmethod
    def load_params_csv(csv_path, corner_id=None):
        df = pd.read_csv(csv_path)
        if corner_id is not None:
            df = df[df['corner_id'] == corner_id]
        row = df.iloc[0]
        return QuarterCarParams(
            m_s=float(row['m_s']),
            m_u=float(row['m_u']),
            k_spring=float(row['k_spring']),
            c_bump=float(row['c_bump']),
            c_rebound=float(row['c_rebound']),
            k_tire=float(row['k_tire']),
            bumpstop_gap=float(row.get('bumpstop_gap', 0.0)),
            bumpstop_rate=float(row.get('bumpstop_rate', 0.0)),
            tire_nl_knee=float(row.get('tire_nl_knee', 0.0)),
            tire_nl_gain=float(row.get('tire_nl_gain', 0.0)),
        )
