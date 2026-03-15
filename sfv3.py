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
# Default fit window (half-width around Ecorr0)
TAFEL_WINDOW_V = 0.30

# Window sensitivity sweep list (values larger than available half-span are auto-ignored per file)
WINDOW_SWEEP_V = [0.15, 0.20, 0.25, 0.30, 0.35]

# Exclude |eta| near 0 for log-domain metrics stability
ETA_EXCLUDE_V = 0.010

# Rp extraction range for experimental slope (±25 mV)
RP_RANGE_V = 0.025

# =====================
# Method display vs filename tags
# =====================
METHODS = {
    "linlin": {"label": "LinC/LinA", "tag": "LinLin"},
    "linnl":  {"label": "LinC/NlA",  "tag": "LinNl"},
    "nlnl":   {"label": "NlC/NlA",   "tag": "NlNl"},
}

def method_label(key: str) -> str:
    return METHODS.get(key, {}).get("label", key)

def method_tag(key: str) -> str:
    return METHODS.get(key, {}).get("tag", key)

def safe_stub(s: str) -> str:
    """
    Make a filesystem-safe stub for filenames. (Does NOT change folder names.)
    Avoids path separators and a few problematic characters.
    """
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_")
    s = s.replace(":", "_")
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

def bv_metrics_on_log_abs(E: np.ndarray, I: np.ndarray, Ecorr: float, i0: float, alpha_a: float, alpha_c: float) -> tuple[float, float]:
    """R² and RMSE computed on log10(|I|) in A, excluding points too close to eta=0."""
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
    """
    Experimental Rp estimate: fit E = Rp*I + b within ±halfwin_v of Ecorr.
    Uses the FULL (smoothed) dataset so the Rp estimate isn't artificially limited by the Tafel window.
    """
    if not np.isfinite(Ecorr):
        return np.nan
    m = np.isfinite(E_all) & np.isfinite(I_all) & (np.abs(E_all - Ecorr) <= halfwin_v)
    if m.sum() < 8:
        return np.nan

    Iw = I_all[m]
    Ew = E_all[m]
    if np.nanstd(Iw) < 1e-12:
        return np.nan

    # E = Rp*I + b  -> Rp in ohm (V/A)
    Rp, _b = np.polyfit(Iw, Ew, 1)
    return float(Rp)

def rp_pred_from_params(beta_a_mV: float, beta_c_mV: float, icorr_A: float) -> float:
    """
    Stern–Geary: Rp = B / icorr, with
    B = (beta_a * beta_c) / (2.303*(beta_a + beta_c)), betas in V/dec
    """
    if not (np.isfinite(beta_a_mV) and np.isfinite(beta_c_mV) and np.isfinite(icorr_A) and icorr_A > 0):
        return np.nan
    beta_a = beta_a_mV / 1000.0
    beta_c = beta_c_mV / 1000.0
    if beta_a <= 0 or beta_c <= 0:
        return np.nan
    B = (beta_a * beta_c) / (2.303 * (beta_a + beta_c))
    return float(B / icorr_A)

# =====================
# Model fits that include Ecorr
# =====================
def fit_lin_lin_ls(E_c, logI_c, E_a, logI_a, Ecorr0, E_bounds):
    """Fit Ecorr, log_i0, slope_a, slope_c to Tafel-domain piecewise linear model."""
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
        "Ecorr": Ecorr,
        "log_i0": log_i0,
        "i0": i0,
        "slope_a": slope_a,
        "slope_c": slope_c,
        "alpha_a": alpha_a,
        "alpha_c": alpha_c,
        "beta_a": beta_a,
        "beta_c": beta_c,
        "lsq": res,
    }

def fit_lin_nl_ls(E_c, logI_c, E_a, logI_a, Ecorr0, E_bounds):
    """Fit Ecorr, log_i0, slope_c (linear cathodic), alpha_a (anodic exponential) in Tafel domain."""
    Emin, Emax = E_bounds
    coeff = F / (2.303 * R * T)

    def resid(p):
        Ec, log_i0, slope_c, alpha_a = p
        eta_c = E_c - Ec
        eta_a = E_a - Ec

        mc = np.abs(eta_c) >= ETA_EXCLUDE_V
        ma = np.abs(eta_a) >= ETA_EXCLUDE_V

        pred_c = slope_c * eta_c + log_i0
        pred_a = log_i0 + alpha_a * coeff * eta_a

        r_c = pred_c - logI_c
        r_a = pred_a - logI_a

        r_c = np.where(mc, r_c, 0.0)
        r_a = np.where(ma, r_a, 0.0)
        return np.concatenate([r_c, r_a])

    log_i0_0 = float(np.nanmedian(np.concatenate([logI_c, logI_a])))
    p0 = np.array([Ecorr0, log_i0_0, -10.0, 0.5], dtype=float)

    lb = np.array([Emin, log_i0_0 - 6.0, -500.0, 0.01], dtype=float)
    ub = np.array([Emax, log_i0_0 + 6.0, -1e-3, 1.0], dtype=float)

    res = least_squares(resid, p0, bounds=(lb, ub),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    Ecorr, log_i0, slope_c, alpha_a = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    alpha_c = -slope_c * 2.303 * R * T / F

    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        "Ecorr": Ecorr,
        "log_i0": log_i0,
        "i0": i0,
        "slope_c": slope_c,
        "alpha_a": alpha_a,
        "alpha_c": alpha_c,
        "beta_a": beta_a,
        "beta_c": beta_c,
        "lsq": res,
    }

def fit_nl_nl_ls(E_all, logI_all, Ecorr0, E_bounds):
    """Fit Ecorr, log_i0, alpha_a, alpha_c by minimizing residuals on log10(|I|) using full BV."""
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
        "Ecorr": Ecorr,
        "log_i0": log_i0,
        "i0": i0,
        "alpha_a": alpha_a,
        "alpha_c": alpha_c,
        "beta_a": beta_a,
        "beta_c": beta_c,
        "lsq": res,
    }

# =====================
# Plot: BV decomposition over LSV data in E-space (I vs E)
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

# =====================
# Utility
# =====================
def corr_rate(i0_A: float) -> float:
    if not (np.isfinite(i0_A) and i0_A > 0):
        return np.nan
    return float(i0_A * K * EW / (rho * A_cm2))

def run_fits_for_window(E_all: np.ndarray, I_sm_all: np.ndarray, Ecorr0: float, window_v: float):
    """
    Run Lin-Lin, Lin-Nl, Nl-Nl on data within +/- window_v around Ecorr0.
    Returns dict or None.
    """
    mwin = (E_all >= Ecorr0 - window_v) & (E_all <= Ecorr0 + window_v)
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
    linnl  = fit_lin_nl_ls(E_c, logI_c, E_a, logI_a, linlin["Ecorr"], E_bounds)
    nlnl   = fit_nl_nl_ls(E_fit, logI_fit, linlin["Ecorr"], E_bounds)

    # Metrics on log(|I|) over the FIT window
    r2_ll, rmse_ll = bv_metrics_on_log_abs(E_fit, I_fit, linlin["Ecorr"], linlin["i0"], linlin["alpha_a"], linlin["alpha_c"])
    r2_ln, rmse_ln = bv_metrics_on_log_abs(E_fit, I_fit, linnl["Ecorr"],  linnl["i0"],  linnl["alpha_a"],  linnl["alpha_c"])
    r2_nn, rmse_nn = bv_metrics_on_log_abs(E_fit, I_fit, nlnl["Ecorr"],   nlnl["i0"],   nlnl["alpha_a"],   nlnl["alpha_c"])

    # Rates
    rate_ll = corr_rate(linlin["i0"])
    rate_ln = corr_rate(linnl["i0"])
    rate_nn = corr_rate(nlnl["i0"])

    # Rp_exp from FULL dataset near each model Ecorr; Rp_pred from SG using that model
    def rp_pair(md: dict) -> tuple[float, float]:
        Ec = md.get("Ecorr", np.nan)
        ic = md.get("i0", np.nan)
        ba = md.get("beta_a", np.nan)
        bc = md.get("beta_c", np.nan)
        rp_exp = rp_exp_from_lsv(E_all, I_sm_all, Ec, halfwin_v=RP_RANGE_V)
        rp_pred = rp_pred_from_params(ba, bc, ic)
        return (rp_exp, rp_pred)

    rp_ll = rp_pair(linlin)
    rp_ln = rp_pair(linnl)
    rp_nn = rp_pair(nlnl)

    return {
        "E_fit": E_fit, "I_fit": I_fit,
        "linlin": linlin, "linnl": linnl, "nlnl": nlnl,
        "metrics": {"linlin": (r2_ll, rmse_ll), "linnl": (r2_ln, rmse_ln), "nlnl": (r2_nn, rmse_nn)},
        "rates": {"linlin": rate_ll, "linnl": rate_ln, "nlnl": rate_nn},
        "rp": {"linlin": rp_ll, "linnl": rp_ln, "nlnl": rp_nn},
    }

# =====================
# Main
# =====================
def main():
    rows_default = []
    rows_sweep = []

    for path in glob.glob("*.csv"):
        if "tafel" in path.lower():
            continue

        out_prefix = os.path.splitext(os.path.basename(path))[0]
        out_prefix_safe = safe_stub(out_prefix)

        plot_dir = out_prefix + "_plots"
        os.makedirs(plot_dir, exist_ok=True)

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not {"Step number", "Working Electrode (V)", "Current (A)"}.issubset(df.columns):
            continue

        data = df[df["Step number"] == 3]
        if len(data) < 20:
            continue

        # Load and smooth (A internally)
        E = data["Working Electrode (V)"].to_numpy(dtype=float)
        I_raw = data["Current (A)"].to_numpy(dtype=float)
        I_sm = optimize_savgol_smoothing(I_raw)

        # Initial Ecorr guess: min abs current point
        Ecorr0 = float(E[np.argmin(np.abs(I_sm))])

        # Determine max half-span available around Ecorr0 in this dataset
        max_halfspan = float(min(np.nanmax(E) - Ecorr0, Ecorr0 - np.nanmin(E)))

        # Windows to run for this file (drop any that exceed available half-span)
        windows_this_file = [float(w) for w in WINDOW_SWEEP_V if float(w) <= (max_halfspan + 1e-12)]
        if TAFEL_WINDOW_V <= max_halfspan + 1e-12 and (TAFEL_WINDOW_V not in windows_this_file):
            windows_this_file.append(float(TAFEL_WINDOW_V))
        windows_this_file = sorted(set(windows_this_file))

        # ---------------------
        # Default run
        # ---------------------
        fit_default = run_fits_for_window(E, I_sm, Ecorr0, TAFEL_WINDOW_V)
        if fit_default is None:
            continue

        # Full-E decomposition plots (use full dataset for the visual overlay)
        for key in ["linlin", "linnl", "nlnl"]:
            md = fit_default[key]
            plot_decomp_over_lsv_E(
                E, I_sm, md,
                os.path.join(plot_dir, f"{out_prefix_safe}_{method_tag(key)}_decomp_E_full.png"),
                title=f"{path}: {method_label(key)} BV Decomp (full E)"
            )

        def add_row(rows_list, window_v: float, method: str, md: dict, r2v: float, rmsev: float, ratev: float, rp_exp: float, rp_pred: float):
            rows_list.append({
                "Filename": path,
                "Window (V)": window_v,
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
        for key in ["linlin", "linnl", "nlnl"]:
            md = fit_default[key]
            r2v, rmsev = fit_default["metrics"][key]
            ratev = fit_default["rates"][key]
            rp_exp, rp_pred = fit_default["rp"][key]
            add_row(rows_default, TAFEL_WINDOW_V, method_label(key), md, r2v, rmsev, ratev, rp_exp, rp_pred)

        # ---------------------
        # Window sensitivity sweep + per-window decomposition plots (windowed data)
        # ---------------------
        sweep_records = []  # for sensitivity plots

        for w in windows_this_file:
            fit_w = run_fits_for_window(E, I_sm, Ecorr0, w)
            if fit_w is None:
                for key in ["linlin", "linnl", "nlnl"]:
                    sweep_records.append((w, method_label(key), np.nan, np.nan))
                continue

            E_fit = fit_w["E_fit"]
            I_fit = fit_w["I_fit"]
            w_mV = int(round(w * 1000))

            for key in ["linlin", "linnl", "nlnl"]:
                md = fit_w[key]
                r2v, rmsev = fit_w["metrics"][key]
                ratev = fit_w["rates"][key]
                rp_exp, rp_pred = fit_w["rp"][key]

                add_row(rows_sweep, w, method_label(key), md, r2v, rmsev, ratev, rp_exp, rp_pred)

                # Decomposition overlay limited to the data USED in this window
                outname = f"{out_prefix_safe}_{method_tag(key)}_decomp_E_win{w_mV}mV.png"
                plot_decomp_over_lsv_E(
                    E_fit, I_fit, md,
                    os.path.join(plot_dir, outname),
                    title=f"{path}: {method_label(key)} BV Decomp (±{w_mV} mV window)"
                )

                sweep_records.append((w, method_label(key), md.get("Ecorr", np.nan), md.get("i0", np.nan) * 1e6))

        # ---------------------
        # Per-file window sensitivity plots
        # ---------------------
        if len(sweep_records) > 0:
            sw_df = pd.DataFrame(sweep_records, columns=["Window (V)", "Method", "Ecorr (V)", "Icorr (µA)"])

            # Icorr vs window
            plt.figure(figsize=(7, 4))
            for meth in sw_df["Method"].unique():
                d = sw_df[sw_df["Method"] == meth].sort_values("Window (V)")
                plt.plot(d["Window (V)"], d["Icorr (µA)"], marker="o", label=meth)
            plt.xlabel("Half-window (V)")
            plt.ylabel("Icorr (µA)")
            plt.title(f"{out_prefix}: Window Sensitivity (Icorr)")
            plt.grid(True)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix_safe}_window_sensitivity_Icorr.png"), dpi=150)
            plt.close()

            # Ecorr vs window
            plt.figure(figsize=(7, 4))
            for meth in sw_df["Method"].unique():
                d = sw_df[sw_df["Method"] == meth].sort_values("Window (V)")
                plt.plot(d["Window (V)"], d["Ecorr (V)"], marker="o", label=meth)
            plt.xlabel("Half-window (V)")
            plt.ylabel("Ecorr (V)")
            plt.title(f"{out_prefix}: Window Sensitivity (Ecorr)")
            plt.grid(True)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix_safe}_window_sensitivity_Ecorr.png"), dpi=150)
            plt.close()

    # ---------------------
    # Outputs
    # ---------------------
    default_df = pd.DataFrame(rows_default)
    if len(default_df) == 0:
        print("Empty summary. No valid CSVs processed.")
        return

    print(default_df.to_string(index=False))
    default_df.to_csv("tafel_summary.csv", index=False)
    print("✅ Default summary saved: tafel_summary.csv")

    sweep_df = pd.DataFrame(rows_sweep)
    if len(sweep_df) > 0:
        sweep_df.to_csv("tafel_window_sensitivity.csv", index=False)
        print("✅ Window sensitivity table saved: tafel_window_sensitivity.csv")

if __name__ == "__main__":
    main()
