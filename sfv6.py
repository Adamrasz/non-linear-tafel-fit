import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# Constants
# =====================
F = 96485    # C/mol
R = 8.314    # J/mol/K
T = 298      # K

# Surface area and corrosion conversion
A_cm2 = 0.3167   # cm^2
EW = 7.021       # g/eq
rho = 2.33       # g/cm^3
K = 3272         # mm·g/(C·cm^3·year)

# =====================
# User controls
# =====================
TAFEL_WINDOW_V = 0.30
WINDOW_SWEEP_V = [0.15, 0.20, 0.25, 0.30, 0.35]   # requested half-windows (V)
ETA_EXCLUDE_V = 0.001                             # exclude |eta| < this in log-domain residuals
RP_RANGE_V = 0.025                                # +/- 25 mV around Ecorr for Rp_exp

VERBOSE = True

# =====================
# Methods (ONLY TWO)
# =====================
METHODS = {
    "linlin": {"label": "LinC/LinA", "tag": "LinLin"},
    "nlnl":   {"label": "NlC/NlA",   "tag": "NlNl"},
}

def method_label(key: str) -> str:
    return METHODS.get(key, {}).get("label", key)

def method_tag(key: str) -> str:
    return METHODS.get(key, {}).get("tag", key)

def safe_stub(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =====================
# Smoothing
# =====================
def smoothness_metric(y: np.ndarray) -> float:
    if len(y) < 3:
        return float("inf")
    return float(np.sum(np.abs(np.diff(y, n=2))))

def optimize_savgol_smoothing(I_raw: np.ndarray) -> np.ndarray:
    best_score = np.inf
    best_sm = I_raw
    for wl in range(11, 101, 2):
        for po in (2, 3):
            if po >= wl:
                continue
            try:
                I_sm = savgol_filter(I_raw, wl, po)
            except Exception:
                continue
            rms = np.sqrt(np.mean((I_raw - I_sm) ** 2))
            curv = smoothness_metric(I_sm)
            score = rms + 1e-3 * curv
            if score < best_score:
                best_score = score
                best_sm = I_sm
    return best_sm

# =====================
# BV / log helpers
# =====================
def safe_log10_abs(I: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    return np.log10(np.maximum(np.abs(I), floor))

def bv_current(eta: np.ndarray, i0: float, alpha_a: float, alpha_c: float) -> np.ndarray:
    arg_a = np.clip(alpha_a * F * eta / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta / (R * T), -50, 50)
    return i0 * (np.exp(arg_a) - np.exp(-arg_c))

def log_bv(eta: np.ndarray, i0: float, alpha_a: float, alpha_c: float) -> np.ndarray:
    return safe_log10_abs(bv_current(eta, i0, alpha_a, alpha_c), floor=1e-20)

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(y_true) < 2:
            return float("nan")
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            return float("nan")
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")

def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(y_true) < 1:
            return float("nan")
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            return float("nan")
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float("nan")

def bv_metrics_on_log_abs(E: np.ndarray, I: np.ndarray, Ecorr: float, i0: float, alpha_a: float, alpha_c: float):
    eta = E - Ecorr
    m = np.isfinite(E) & np.isfinite(I) & (np.abs(eta) >= ETA_EXCLUDE_V)
    if m.sum() < 3:
        return (np.nan, np.nan)
    y = safe_log10_abs(I[m])
    yhat = log_bv(eta[m], i0, alpha_a, alpha_c)
    return (_safe_r2(y, yhat), _safe_rmse(y, yhat))

# =====================
# Rp helpers
# =====================
def rp_exp_from_lsv(E_all: np.ndarray, I_all: np.ndarray, Ecorr: float, halfwin_v: float = RP_RANGE_V) -> float:
    if not np.isfinite(Ecorr):
        return np.nan
    m = np.isfinite(E_all) & np.isfinite(I_all) & (np.abs(E_all - Ecorr) <= halfwin_v)
    if m.sum() < 8:
        return np.nan
    Iw = I_all[m]
    Ew = E_all[m]
    if np.nanstd(Iw) < 1e-12:
        return np.nan
    Rp, _b = np.polyfit(Iw, Ew, 1)  # E = Rp*I + b
    return float(Rp)

def rp_pred_from_params(beta_a_mV: float, beta_c_mV: float, icorr_A: float) -> float:
    if not (np.isfinite(beta_a_mV) and np.isfinite(beta_c_mV) and np.isfinite(icorr_A) and icorr_A > 0):
        return np.nan
    beta_a = beta_a_mV / 1000.0
    beta_c = beta_c_mV / 1000.0
    if beta_a <= 0 or beta_c <= 0:
        return np.nan
    B = (beta_a * beta_c) / (2.303 * (beta_a + beta_c))
    return float(B / icorr_A)

# =====================
# Artifact handling / Ecorr0 estimate
# =====================
def drop_initial_outlier(E: np.ndarray, I: np.ndarray, k: int = 10, z: float = 8.0):
    """
    Drop the first point if it's a strong outlier relative to the next k points.
    This targets the common "first point glitch" at scan start.
    """
    if len(I) < k + 2:
        return E, I, False

    ref = I[1:k+1]
    medI = float(np.nanmedian(ref))
    medAbs = float(np.nanmedian(np.abs(ref))) + 1e-30

    # sign flip + huge magnitude compared to typical early points
    if np.sign(I[0]) != np.sign(medI) and (np.abs(I[0]) > 10.0 * medAbs):
        return E[1:], I[1:], True

    # log-domain deviation test
    ref_log = safe_log10_abs(ref)
    med = float(np.nanmedian(ref_log))
    mad = float(np.nanmedian(np.abs(ref_log - med))) + 1e-12
    x0 = float(safe_log10_abs(np.array([I[0]]))[0])

    if np.isfinite(x0) and np.abs(x0 - med) > z * mad:
        return E[1:], I[1:], True

    return E, I, False

def estimate_ecorr_zero_cross(E: np.ndarray, I: np.ndarray) -> float:
    """
    Prefer a sign-change (I crosses 0) interpolation; fallback to min |I|.
    Uses the data in its recorded order (not sorted).
    """
    if len(E) < 3:
        return float(E[np.argmin(np.abs(I))])

    s = np.sign(I)
    # find indices where sign changes between consecutive points
    idx = np.where((s[:-1] * s[1:]) < 0)[0]
    if len(idx) > 0:
        # choose the sign-change pair with smallest |I| near crossing
        best = None
        for i in idx:
            I1, I2 = I[i], I[i+1]
            if not (np.isfinite(I1) and np.isfinite(I2)):
                continue
            denom = (I2 - I1)
            if denom == 0:
                continue
            # linear interpolation to I=0
            E1, E2 = E[i], E[i+1]
            Ec = E1 + (0.0 - I1) * (E2 - E1) / denom
            score = min(abs(I1), abs(I2))
            if (best is None) or (score < best[0]):
                best = (score, float(Ec))
        if best is not None:
            return best[1]

    return float(E[np.argmin(np.abs(I))])

# =====================
# Model fits (Ecorr included)
# =====================
def fit_lin_lin_ls(E_c, logI_c, E_a, logI_a, Ecorr0, E_bounds):
    Emin, Emax = E_bounds

    def resid(p):
        Ec, log_i0, slope_a, slope_c = p
        eta_c = E_c - Ec
        eta_a = E_a - Ec
        mc = np.abs(eta_c) >= ETA_EXCLUDE_V
        ma = np.abs(eta_a) >= ETA_EXCLUDE_V
        r_c = (slope_c * eta_c + log_i0) - logI_c
        r_a = (slope_a * eta_a + log_i0) - logI_a
        r_c = np.where(mc, r_c, 0.0)
        r_a = np.where(ma, r_a, 0.0)
        return np.concatenate([r_c, r_a])

    log_i0_0 = float(np.nanmedian(np.concatenate([logI_c, logI_a])))
    p0 = np.array([Ecorr0, log_i0_0, 10.0, -10.0], dtype=float)

    lb = np.array([Emin, log_i0_0 - 6.0, 1e-3, -500.0], dtype=float)
    ub = np.array([Emax, log_i0_0 + 6.0, 500.0, -1e-3], dtype=float)

    res = least_squares(resid, p0, bounds=(lb, ub),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    Ecorr, log_i0, slope_a, slope_c = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    alpha_a = slope_a * 2.303 * R * T / F
    alpha_c = -slope_c * 2.303 * R * T / F

    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        "Ecorr": Ecorr, "log_i0": log_i0, "i0": i0,
        "slope_a": slope_a, "slope_c": slope_c,
        "alpha_a": alpha_a, "alpha_c": alpha_c,
        "beta_a": beta_a, "beta_c": beta_c,
        "lsq": res,
    }

def fit_nl_nl_ls(E_all, logI_all, Ecorr0, E_bounds):
    Emin, Emax = E_bounds

    def resid(p):
        Ec, log_i0, aa, ac = p
        i0 = 10 ** log_i0
        eta = E_all - Ec
        m = np.abs(eta) >= ETA_EXCLUDE_V
        r = log_bv(eta, i0, aa, ac) - logI_all
        return np.where(m, r, 0.0)

    log_i0_0 = float(np.nanmedian(logI_all))
    p0 = np.array([Ecorr0, log_i0_0, 0.5, 0.5], dtype=float)

    lb = np.array([Emin, log_i0_0 - 8.0, 0.01, 0.01], dtype=float)
    ub = np.array([Emax, log_i0_0 + 8.0, 1.0, 1.0], dtype=float)

    res = least_squares(resid, p0, bounds=(lb, ub),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=50000)
    Ecorr, log_i0, alpha_a, alpha_c = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        "Ecorr": Ecorr, "log_i0": log_i0, "i0": i0,
        "alpha_a": alpha_a, "alpha_c": alpha_c,
        "beta_a": beta_a, "beta_c": beta_c,
        "lsq": res,
    }

# =====================
# Plotting (NO filename in title)
# =====================
def plot_decomp_over_lsv_E(E, I_sm, md, outpath, title):
    Ec = md.get("Ecorr", np.nan)
    i0 = md.get("i0", np.nan)
    aa = md.get("alpha_a", np.nan)
    ac = md.get("alpha_c", np.nan)

    if not (np.isfinite(Ec) and np.isfinite(i0) and np.isfinite(aa) and np.isfinite(ac) and i0 > 0 and aa > 0 and ac > 0):
        return

    E_grid = np.linspace(float(np.nanmin(E)), float(np.nanmax(E)), 800)
    eta = E_grid - Ec

    arg_a = np.clip(aa * F * eta / (R * T), -50, 50)
    arg_c = np.clip(ac * F * eta / (R * T), -50, 50)

    Ia = i0 * np.exp(arg_a)
    Ic = -i0 * np.exp(-arg_c)
    It = Ia + Ic

    plt.figure(figsize=(7, 4))
    plt.plot(E, I_sm * 1e3, label="Smoothed LSV")
    plt.plot(E_grid, It * 1e3, label="I_model (Ia+Ic)")
    plt.plot(E_grid, Ia * 1e3, linestyle="--", label="Ia")
    plt.plot(E_grid, Ic * 1e3, linestyle="--", label="Ic")
    plt.axhline(0, linestyle=":", linewidth=1)
    plt.axvline(Ec, linestyle=":", linewidth=1)
    plt.xlabel("E (V)")
    plt.ylabel("Current (mA)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_tafel_log_overlay(E_fit, I_fit, linlin, nlnl, Ecorr_ref, outpath, title):
    eta_ref = E_fit - Ecorr_ref
    logI_mA = safe_log10_abs(I_fit) + 3.0  # A -> mA
    sidx = np.argsort(eta_ref)

    E_grid = np.linspace(float(np.min(E_fit)), float(np.max(E_fit)), 500)
    eta_grid_ref = E_grid - Ecorr_ref

    plt.figure(figsize=(7, 4))
    plt.plot(eta_ref[sidx], logI_mA[sidx], "k-", label="Data")

    Ec = linlin.get("Ecorr", np.nan)
    if np.isfinite(Ec):
        y = log_bv(E_grid - Ec, linlin["i0"], linlin["alpha_a"], linlin["alpha_c"]) + 3.0
        plt.plot(eta_grid_ref, y, label="Lin-Lin (BV)")

    Ec = nlnl.get("Ecorr", np.nan)
    if np.isfinite(Ec):
        y = log_bv(E_grid - Ec, nlnl["i0"], nlnl["alpha_a"], nlnl["alpha_c"]) + 3.0
        plt.plot(eta_grid_ref, y, label="Nl-Nl (BV)")

    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("η (V)")
    plt.ylabel("log10(|I|) (mA)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_linlin_baselines(E_fit, I_fit, linlin, outpath, title):
    eta = E_fit - linlin["Ecorr"]
    logI_mA = safe_log10_abs(I_fit) + 3.0
    sidx = np.argsort(eta)

    log_i0 = linlin["log_i0"]
    slope_a = linlin["slope_a"]
    slope_c = linlin["slope_c"]

    x_c = np.linspace(float(np.min(eta)), 0.0, 200)
    x_a = np.linspace(0.0, float(np.max(eta)), 200)

    plt.figure(figsize=(7, 4))
    plt.plot(eta[sidx], logI_mA[sidx], "k-", label="Data")
    plt.plot(x_c, slope_c * x_c + log_i0 + 3.0, "b--", label=f"Cath lin (βc={linlin['beta_c']:.1f} mV/dec)")
    plt.plot(x_a, slope_a * x_a + log_i0 + 3.0, "r--", label=f"Anod lin (βa={linlin['beta_a']:.1f} mV/dec)")

    plt.axvline(0, linestyle="--", color="gray")
    plt.axhline(log_i0 + 3.0, linestyle=":", color="gray")
    plt.xlabel("η (V)")
    plt.ylabel("log10(|I|) (mA)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =====================
# Utility
# =====================
def corr_rate(i0_A: float) -> float:
    if not (np.isfinite(i0_A) and i0_A > 0):
        return np.nan
    return float(i0_A * K * EW / (rho * A_cm2))

def run_fits_for_window_asym(E_all: np.ndarray, I_sm_all: np.ndarray, Ecorr0: float, win_neg: float, win_pos: float):
    """
    Asymmetric window around Ecorr0:
      cathodic side uses win_neg
      anodic side uses win_pos
    Selection is E in [Ecorr0 - win_neg, Ecorr0 + win_pos].
    """
    if not (np.isfinite(win_neg) and np.isfinite(win_pos)) or win_neg <= 0 or win_pos <= 0:
        return None

    mwin = (E_all >= Ecorr0 - win_neg) & (E_all <= Ecorr0 + win_pos)
    if mwin.sum() < 20:
        return None

    E_fit = E_all[mwin]
    I_fit = I_sm_all[mwin]
    logI_fit = safe_log10_abs(I_fit)

    mc = I_fit < 0
    ma = I_fit > 0
    if mc.sum() < 8 or ma.sum() < 8:
        return None

    E_c = E_fit[mc]
    E_a = E_fit[ma]
    logI_c = safe_log10_abs(I_fit[mc])
    logI_a = safe_log10_abs(I_fit[ma])

    E_bounds = (float(E_fit.min()), float(E_fit.max()))

    linlin = fit_lin_lin_ls(E_c, logI_c, E_a, logI_a, Ecorr0, E_bounds)
    nlnl   = fit_nl_nl_ls(E_fit, logI_fit, linlin["Ecorr"], E_bounds)

    r2_ll, rmse_ll = bv_metrics_on_log_abs(E_fit, I_fit, linlin["Ecorr"], linlin["i0"], linlin["alpha_a"], linlin["alpha_c"])
    r2_nn, rmse_nn = bv_metrics_on_log_abs(E_fit, I_fit, nlnl["Ecorr"],   nlnl["i0"],   nlnl["alpha_a"],   nlnl["alpha_c"])

    rate_ll = corr_rate(linlin["i0"])
    rate_nn = corr_rate(nlnl["i0"])

    def rp_pair(md: dict) -> tuple[float, float]:
        Ec = md.get("Ecorr", np.nan)
        ic = md.get("i0", np.nan)
        ba = md.get("beta_a", np.nan)
        bc = md.get("beta_c", np.nan)
        rp_exp = rp_exp_from_lsv(E_all, I_sm_all, Ec, halfwin_v=RP_RANGE_V)
        rp_pred = rp_pred_from_params(ba, bc, ic)
        return (rp_exp, rp_pred)

    rp_ll = rp_pair(linlin)
    rp_nn = rp_pair(nlnl)

    return {
        "E_fit": E_fit, "I_fit": I_fit,
        "linlin": linlin, "nlnl": nlnl,
        "metrics": {"linlin": (r2_ll, rmse_ll), "nlnl": (r2_nn, rmse_nn)},
        "rates": {"linlin": rate_ll, "nlnl": rate_nn},
        "rp": {"linlin": rp_ll, "nlnl": rp_nn},
    }

# =====================
# Main
# =====================
def main():
    rows_default = []
    rows_sweep = []

    generated = {"tafel_summary.csv", "tafel_window_sensitivity.csv"}
    paths = sorted(set(glob.glob("*.csv") + glob.glob("*.CSV")))

    if VERBOSE:
        print(f"Found {len(paths)} CSV files in folder.")

    for path in paths:
        if os.path.basename(path) in generated:
            continue

        out_prefix = os.path.splitext(os.path.basename(path))[0]
        out_prefix_safe = safe_stub(out_prefix)
        plot_dir = out_prefix + "_plots"
        os.makedirs(plot_dir, exist_ok=True)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            if VERBOSE:
                print(f"[SKIP] {path}: read_csv failed -> {e}")
            continue

        # normalize column names (does not broaden schema; just avoids whitespace mismatches)
        df.columns = [str(c).strip() for c in df.columns]

        required = {"Step number", "Working Electrode (V)", "Current (A)"}
        if not required.issubset(df.columns):
            if VERBOSE:
                missing = sorted(required - set(df.columns))
                print(f"[SKIP] {path}: missing columns {missing}")
            continue

        # coerce to numeric
        step = pd.to_numeric(df["Step number"], errors="coerce").to_numpy(dtype=float)
        E_all = pd.to_numeric(df["Working Electrode (V)"], errors="coerce").to_numpy(dtype=float)
        I_all = pd.to_numeric(df["Current (A)"], errors="coerce").to_numpy(dtype=float)

        # ---- auto-select step (prefers 3, else 2) ----
        candidates = [3, 2]
        best_step = None
        best_mask = None
        best_n = -1

        for s in candidates:
            m = (step == s) & np.isfinite(E_all) & np.isfinite(I_all)
            n = int(np.sum(m))
            if n > best_n:
                best_n = n
                best_step = s
                best_mask = m

        if best_step is None or best_n < 20:
            if VERBOSE:
                uniq = sorted([int(x) for x in pd.Series(step).dropna().unique()])
                print(f"[SKIP] {path}: neither Step 3 nor 2 has >=20 finite rows. Steps found={uniq}, best={best_step} n={best_n}")
            continue

        E = E_all[best_mask]
        I_raw = I_all[best_mask]

        # ---- drop start-of-scan artifact if present ----
        E, I_raw, dropped = drop_initial_outlier(E, I_raw, k=10, z=8.0)
        if dropped and VERBOSE:
            print(f"[INFO] {path}: dropped initial outlier point in Step {best_step}")

        if len(E) < 20:
            if VERBOSE:
                print(f"[SKIP] {path}: insufficient points after outlier drop.")
            continue

        I_sm = optimize_savgol_smoothing(I_raw)

        # Ecorr0 anchored to zero-crossing when possible
        Ecorr0 = estimate_ecorr_zero_cross(E, I_sm)

        # Determine available span per branch (based on sign of smoothed current)
        mc_all = I_sm < 0
        ma_all = I_sm > 0
        if not mc_all.any() or not ma_all.any():
            if VERBOSE:
                print(f"[SKIP] {path}: missing one branch after smoothing (need both I<0 and I>0).")
            continue

        cath_span = float(Ecorr0 - np.nanmin(E[mc_all]))
        anod_span = float(np.nanmax(E[ma_all]) - Ecorr0)

        if cath_span <= 0.02 or anod_span <= 0.02:
            if VERBOSE:
                print(f"[SKIP] {path}: insufficient span (cath_span={cath_span:.3f} V, anod_span={anod_span:.3f} V).")
            continue

        # Windows to attempt (meaningful only up to anod_span; cath side will clip automatically)
        windows_this_file = [float(w) for w in WINDOW_SWEEP_V if float(w) <= (anod_span + 1e-12)]
        if TAFEL_WINDOW_V <= anod_span + 1e-12 and (TAFEL_WINDOW_V not in windows_this_file):
            windows_this_file.append(float(TAFEL_WINDOW_V))
        windows_this_file = sorted(set(windows_this_file))

        if len(windows_this_file) == 0:
            if VERBOSE:
                print(f"[SKIP] {path}: no requested windows fit anodic span (anod_span={anod_span:.3f} V).")
            continue

        # ---------------------
        # Default window (asymmetric)
        # ---------------------
        w_req = float(TAFEL_WINDOW_V)
        w_neg = min(w_req, cath_span)
        w_pos = min(w_req, anod_span)

        fit_default = run_fits_for_window_asym(E, I_sm, Ecorr0, w_neg, w_pos)
        if fit_default is None:
            if VERBOSE:
                print(f"[SKIP] {path}: default asym window failed (neg={w_neg:.3f}, pos={w_pos:.3f}).")
            continue

        if VERBOSE:
            print(f"[OK] {path}: Step {best_step} | cath_span={cath_span:.3f} V, anod_span={anod_span:.3f} V | default (neg={w_neg:.3f}, pos={w_pos:.3f})")

        linlin = fit_default["linlin"]
        nlnl = fit_default["nlnl"]
        E_fit = fit_default["E_fit"]
        I_fit = fit_default["I_fit"]

        Ecorr_ref = nlnl["Ecorr"] if np.isfinite(nlnl.get("Ecorr", np.nan)) else linlin["Ecorr"]

        # Full-E decomposition (no filename in title)
        for key in ["linlin", "nlnl"]:
            md = fit_default[key]
            title = f"{method_label(key)} — BV Decomposition (full E)"
            plot_decomp_over_lsv_E(
                E, I_sm, md,
                os.path.join(plot_dir, f"{out_prefix_safe}_{method_tag(key)}_decomp_E_full.png"),
                title=title
            )

        # Log overlay (default)
        plot_tafel_log_overlay(
            E_fit, I_fit, linlin, nlnl, Ecorr_ref,
            os.path.join(plot_dir, f"{out_prefix_safe}_tafel_overlay_log_default.png"),
            title=f"Tafel Overlay (log|I|, default)"
        )

        # Linlin baselines (default)
        plot_linlin_baselines(
            E_fit, I_fit, linlin,
            os.path.join(plot_dir, f"{out_prefix_safe}_tafel_linear_baselines_default.png"),
            title=f"Lin-Lin Baselines (default)"
        )

        def add_row(rows_list, win_req: float, win_neg: float, win_pos: float, method: str, md: dict,
                    r2v: float, rmsev: float, ratev: float, rp_exp: float, rp_pred: float):
            rows_list.append({
                "Filename": path,
                "Window_req (V)": win_req,
                "Window_cath_used (V)": win_neg,
                "Window_anod_used (V)": win_pos,
                "Method": method,
                "Ecorr (V)": md.get("Ecorr", np.nan),
                "βc (mV/dec)": md.get("beta_c", np.nan),
                "βa (mV/dec)": md.get("beta_a", np.nan),
                "αc": md.get("alpha_c", np.nan),
                "αa": md.get("alpha_a", np.nan),
                "Icorr (µA)": (md.get("i0", np.nan) * 1e6) if np.isfinite(md.get("i0", np.nan)) else np.nan,
                "Rate (mmpy)": ratev,
                "R² (BV on log)": r2v,
                "RMSE (BV on log)": rmsev,
                "Rp_exp (ohm)": rp_exp,
                "Rp_pred (ohm)": rp_pred,
                "Rp_ratio (exp/pred)": (rp_exp / rp_pred) if (np.isfinite(rp_exp) and np.isfinite(rp_pred) and rp_pred != 0) else np.nan,
            })

        # Default summary rows
        for key in ["linlin", "nlnl"]:
            md = fit_default[key]
            r2v, rmsev = fit_default["metrics"][key]
            ratev = fit_default["rates"][key]
            rp_exp, rp_pred = fit_default["rp"][key]
            add_row(rows_default, w_req, w_neg, w_pos, method_label(key), md, r2v, rmsev, ratev, rp_exp, rp_pred)

        # ---------------------
        # Window sensitivity sweep (asymmetric)
        # ---------------------
        sweep_records = []
        for w in windows_this_file:
            w = float(w)
            wn = min(w, cath_span)
            wp = min(w, anod_span)

            fit_w = run_fits_for_window_asym(E, I_sm, Ecorr0, wn, wp)
            if fit_w is None:
                for key in ["linlin", "nlnl"]:
                    sweep_records.append((w, wn, wp, method_label(key), np.nan, np.nan))
                continue

            E_w = fit_w["E_fit"]
            I_w = fit_w["I_fit"]
            w_mV = int(round(w * 1000))

            linlin_w = fit_w["linlin"]
            nlnl_w = fit_w["nlnl"]
            Ecorr_ref_w = nlnl_w["Ecorr"] if np.isfinite(nlnl_w.get("Ecorr", np.nan)) else linlin_w["Ecorr"]

            # per-window overlay plots in log space (titles generic)
            plot_tafel_log_overlay(
                E_w, I_w, linlin_w, nlnl_w, Ecorr_ref_w,
                os.path.join(plot_dir, f"{out_prefix_safe}_tafel_overlay_log_win{w_mV}mV.png"),
                title=f"Tafel Overlay (±{w_mV} mV req; cath used ±{int(round(wn*1000))} mV)"
            )

            # per-window linlin baselines
            plot_linlin_baselines(
                E_w, I_w, linlin_w,
                os.path.join(plot_dir, f"{out_prefix_safe}_tafel_linear_baselines_win{w_mV}mV.png"),
                title=f"Lin-Lin Baselines (±{w_mV} mV req)"
            )

            # per-window decomposition plots
            for key in ["linlin", "nlnl"]:
                md = fit_w[key]
                r2v, rmsev = fit_w["metrics"][key]
                ratev = fit_w["rates"][key]
                rp_exp, rp_pred = fit_w["rp"][key]

                add_row(rows_sweep, w, wn, wp, method_label(key), md, r2v, rmsev, ratev, rp_exp, rp_pred)

                outname = f"{out_prefix_safe}_{method_tag(key)}_decomp_E_win{w_mV}mV.png"
                plot_decomp_over_lsv_E(
                    E_w, I_w, md,
                    os.path.join(plot_dir, outname),
                    title=f"{method_label(key)} — BV Decomposition (windowed)"
                )

                sweep_records.append((w, wn, wp, method_label(key), md.get("Ecorr", np.nan), md.get("i0", np.nan) * 1e6))

        # Window sensitivity plots per file (titles generic)
        if len(sweep_records) > 0:
            sw_df = pd.DataFrame(sweep_records, columns=["Window_req (V)", "Window_cath_used (V)", "Window_anod_used (V)", "Method", "Ecorr (V)", "Icorr (µA)"])

            plt.figure(figsize=(7, 4))
            for meth in sw_df["Method"].unique():
                d = sw_df[sw_df["Method"] == meth].sort_values("Window_req (V)")
                plt.plot(d["Window_req (V)"], d["Icorr (µA)"], marker="o", label=meth)
            plt.xlabel("Requested half-window (V)")
            plt.ylabel("Icorr (µA)")
            plt.title("Window Sensitivity (Icorr)")
            plt.grid(True)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix_safe}_window_sensitivity_Icorr.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(7, 4))
            for meth in sw_df["Method"].unique():
                d = sw_df[sw_df["Method"] == meth].sort_values("Window_req (V)")
                plt.plot(d["Window_req (V)"], d["Ecorr (V)"], marker="o", label=meth)
            plt.xlabel("Requested half-window (V)")
            plt.ylabel("Ecorr (V)")
            plt.title("Window Sensitivity (Ecorr)")
            plt.grid(True)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix_safe}_window_sensitivity_Ecorr.png"), dpi=150)
            plt.close()

    # =====================
    # Save outputs (aggregate across all processed files in this run)
    # =====================
    default_df = pd.DataFrame(rows_default)
    if len(default_df) == 0:
        print("Empty summary. No valid CSVs processed.")
        return

    print(default_df.to_string(index=False))
    default_df.to_csv("tafel_summary.csv", index=False)
    print("✅ Saved: tafel_summary.csv")

    sweep_df = pd.DataFrame(rows_sweep)
    if len(sweep_df) > 0:
        sweep_df.to_csv("tafel_window_sensitivity.csv", index=False)
        print("✅ Saved: tafel_window_sensitivity.csv")

if __name__ == "__main__":
    main()
