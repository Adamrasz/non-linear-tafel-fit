import os
import glob
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
# Main fit window (default run + plots)
TAFEL_WINDOW_V = 0.30      # +/- window around initial Ecorr guess (in V)

# Window sensitivity sweep (run additional fits without re-smoothing)
WINDOW_SWEEP_V = [0.15, 0.20, 0.25, 0.30, 0.35]

# Exclude |eta| very near 0 to avoid BV log singularity around eta=0
ETA_EXCLUDE_V = 0.010

# Refined Joint model toggle (computationally expensive)
RUN_REFINED_JOINT = False
ECORR_SEARCH_V = 0.050     # refined-joint Ecorr grid half-span (in V)

# =====================
# Smoothing
# =====================
def smoothness_metric(y: np.ndarray) -> float:
    if len(y) < 3:
        return float('inf')
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
            return float('nan')
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            return float('nan')
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float('nan')

def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(y_true) < 1:
            return float('nan')
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            return float('nan')
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float('nan')

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
# Model fits that include Ecorr
# =====================
def fit_lin_lin_ls(
    E_c: np.ndarray,
    logI_c: np.ndarray,
    E_a: np.ndarray,
    logI_a: np.ndarray,
    Ecorr0: float,
    E_bounds: tuple[float, float],
):
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

    res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    Ecorr, log_i0, slope_a, slope_c = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    alpha_a = slope_a * 2.303 * R * T / F
    alpha_c = -slope_c * 2.303 * R * T / F

    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        'Ecorr': Ecorr,
        'log_i0': log_i0,
        'i0': i0,
        'slope_a': slope_a,
        'slope_c': slope_c,
        'alpha_a': alpha_a,
        'alpha_c': alpha_c,
        'beta_a': beta_a,
        'beta_c': beta_c,
        'lsq': res,
    }

def fit_lin_nl_ls(
    E_c: np.ndarray,
    logI_c: np.ndarray,
    E_a: np.ndarray,
    logI_a: np.ndarray,
    Ecorr0: float,
    E_bounds: tuple[float, float],
):
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

    res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    Ecorr, log_i0, slope_c, alpha_a = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    alpha_c = -slope_c * 2.303 * R * T / F

    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        'Ecorr': Ecorr,
        'log_i0': log_i0,
        'i0': i0,
        'slope_c': slope_c,
        'alpha_a': alpha_a,
        'alpha_c': alpha_c,
        'beta_a': beta_a,
        'beta_c': beta_c,
        'lsq': res,
    }

def fit_nl_nl_ls(
    E_all: np.ndarray,
    logI_all: np.ndarray,
    Ecorr0: float,
    E_bounds: tuple[float, float],
):
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

    res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=50000)
    Ecorr, log_i0, alpha_a, alpha_c = [float(x) for x in res.x]

    i0 = 10 ** log_i0
    beta_a = 2.303 * R * T / (alpha_a * F) * 1000 if alpha_a > 0 else np.nan
    beta_c = 2.303 * R * T / (alpha_c * F) * 1000 if alpha_c > 0 else np.nan

    return {
        'Ecorr': Ecorr,
        'log_i0': log_i0,
        'i0': i0,
        'alpha_a': alpha_a,
        'alpha_c': alpha_c,
        'beta_a': beta_a,
        'beta_c': beta_c,
        'lsq': res,
    }

# =====================
# Plot: BV decomposition over LSV data in E-space (I vs E)
# =====================
def plot_decomp_over_lsv_E(
    E: np.ndarray,
    I_sm: np.ndarray,
    md: dict,
    outpath: str,
    title: str,
):
    Ec = md.get('Ecorr', np.nan)
    i0 = md.get('i0', np.nan)
    aa = md.get('alpha_a', np.nan)
    ac = md.get('alpha_c', np.nan)

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
    plt.plot(E, I_sm * 1e3, label='Smoothed LSV')      # mA
    plt.plot(E_grid, It * 1e3, label='I_model (Ia+Ic)')# mA
    plt.plot(E_grid, Ia * 1e3, linestyle='--', label='Ia')
    plt.plot(E_grid, Ic * 1e3, linestyle='--', label='Ic')
    plt.axhline(0, linestyle=':', linewidth=1)
    plt.axvline(Ec, linestyle=':', linewidth=1)
    plt.xlabel('E (V)')
    plt.ylabel('Current (mA)')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =====================
# Helpers
# =====================
def corr_rate(i0_A: float) -> float:
    if not (np.isfinite(i0_A) and i0_A > 0):
        return np.nan
    return float(i0_A * K * EW / (rho * A_cm2))

def run_fits_for_window(E: np.ndarray, I_sm: np.ndarray, Ecorr0: float, window_v: float):
    """Run Lin-Lin, Lin-Nl, Nl-Nl on data within +/- window_v around Ecorr0. Returns dict or None."""
    mwin = (E >= Ecorr0 - window_v) & (E <= Ecorr0 + window_v)
    if mwin.sum() < 20:
        return None

    E_fit = E[mwin]
    I_fit = I_sm[mwin]
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
    linnl  = fit_lin_nl_ls(E_c, logI_c, E_a, logI_a, linlin['Ecorr'], E_bounds)
    nlnl   = fit_nl_nl_ls(E_fit, logI_fit, linlin['Ecorr'], E_bounds)

    # metrics + rates
    r2_ll, rmse_ll = bv_metrics_on_log_abs(E_fit, I_fit, linlin['Ecorr'], linlin['i0'], linlin['alpha_a'], linlin['alpha_c'])
    r2_ln, rmse_ln = bv_metrics_on_log_abs(E_fit, I_fit, linnl['Ecorr'],  linnl['i0'],  linnl['alpha_a'],  linnl['alpha_c'])
    r2_nn, rmse_nn = bv_metrics_on_log_abs(E_fit, I_fit, nlnl['Ecorr'],   nlnl['i0'],   nlnl['alpha_a'],   nlnl['alpha_c'])

    out = {
        'E_fit': E_fit, 'I_fit': I_fit,
        'linlin': linlin, 'linnl': linnl, 'nlnl': nlnl,
        'metrics': {
            'linlin': (r2_ll, rmse_ll),
            'linnl':  (r2_ln, rmse_ln),
            'nlnl':   (r2_nn, rmse_nn),
        },
        'rates': {
            'linlin': corr_rate(linlin['i0']),
            'linnl':  corr_rate(linnl['i0']),
            'nlnl':   corr_rate(nlnl['i0']),
        }
    }
    return out

# =====================
# Main batch processing
# =====================
def main():
    rows_default = []
    rows_sweep = []

    for path in glob.glob('*.csv'):
        if 'tafel' in path.lower():
            continue

        out_prefix = os.path.splitext(os.path.basename(path))[0]
        plot_dir = out_prefix + '_plots'
        os.makedirs(plot_dir, exist_ok=True)

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not {'Step number', 'Working Electrode (V)', 'Current (A)'}.issubset(df.columns):
            continue

        data = df[df['Step number'] == 3]
        if len(data) < 20:
            continue

        # ---------------------
        # Load and smooth (A internally)
        # ---------------------
        E = data['Working Electrode (V)'].to_numpy(dtype=float)
        I_raw = data['Current (A)'].to_numpy(dtype=float)
        I_sm = optimize_savgol_smoothing(I_raw)

        # Initial Ecorr guess: min abs current point
        Ecorr0 = float(E[np.argmin(np.abs(I_sm))])

        # ---------------------
        # Default window run (keeps your usual plots)
        # ---------------------
        fit_default = run_fits_for_window(E, I_sm, Ecorr0, TAFEL_WINDOW_V)
        if fit_default is None:
            continue

        linlin = fit_default['linlin']
        linnl  = fit_default['linnl']
        nlnl   = fit_default['nlnl']

        r2_ll, rmse_ll = fit_default['metrics']['linlin']
        r2_ln, rmse_ln = fit_default['metrics']['linnl']
        r2_nn, rmse_nn = fit_default['metrics']['nlnl']

        rate_linlin = fit_default['rates']['linlin']
        rate_linnl  = fit_default['rates']['linnl']
        rate_nlnl   = fit_default['rates']['nlnl']

        # ---------------------
        # NEW: BV decomposition overlays in E-space
        # ---------------------
        plot_decomp_over_lsv_E(
            E, I_sm, linlin,
            os.path.join(plot_dir, f"{out_prefix}_Lin-Lin_decomp_E.png"),
            title=f"{path}: Lin-Lin BV Decomp (E-space)"
        )
        plot_decomp_over_lsv_E(
            E, I_sm, linnl,
            os.path.join(plot_dir, f"{out_prefix}_Lin-Nl_decomp_E.png"),
            title=f"{path}: Lin-Nl BV Decomp (E-space)"
        )
        plot_decomp_over_lsv_E(
            E, I_sm, nlnl,
            os.path.join(plot_dir, f"{out_prefix}_Nl-Nl_decomp_E.png"),
            title=f"{path}: Nl-Nl BV Decomp (E-space)"
        )

        # ---------------------
        # Default-window summary rows
        # ---------------------
        def add_row(rows_list, method: str, md: dict, r2v: float, rmsev: float, ratev: float, window_v: float):
            rows_list.append({
                'Filename': path,
                'Window (V)': window_v,
                'Method': method,
                'Ecorr (V)': md.get('Ecorr', np.nan),
                'βc (mV/dec)': md.get('beta_c', np.nan),
                'βa (mV/dec)': md.get('beta_a', np.nan),
                'αc': md.get('alpha_c', np.nan),
                'αa': md.get('alpha_a', np.nan),
                'Icorr (µA)': (md.get('i0', np.nan) * 1e6) if np.isfinite(md.get('i0', np.nan)) else np.nan,
                'Rate (mmpy)': ratev,
                'R² (BV on log)': r2v,
                'RMSE (BV on log)': rmsev,
            })

        add_row(rows_default, 'LinC/LinA', linlin, r2_ll, rmse_ll, rate_linlin, TAFEL_WINDOW_V)
        add_row(rows_default, 'LinC/NlA',  linnl,  r2_ln, rmse_ln, rate_linnl,  TAFEL_WINDOW_V)
        add_row(rows_default, 'NlC/NlA',   nlnl,   r2_nn, rmse_nn, rate_nlnl,   TAFEL_WINDOW_V)

        # ---------------------
        # NEW: Window sensitivity sweep
        # ---------------------
        sweep_records = []  # per-file collection for plotting

        for w in WINDOW_SWEEP_V:
            fit_w = run_fits_for_window(E, I_sm, Ecorr0, float(w))
            if fit_w is None:
                # still write NaNs so plots show gaps
                for meth in ['LinC/LinA', 'LinC/NlA', 'NlC/NlA']:
                    sweep_records.append((w, meth, np.nan, np.nan))
                continue

            ll = fit_w['linlin']
            ln = fit_w['linnl']
            nn = fit_w['nlnl']

            r2ll, rmsell = fit_w['metrics']['linlin']
            r2ln, rmseln = fit_w['metrics']['linnl']
            r2nn, rmsenn = fit_w['metrics']['nlnl']

            add_row(rows_sweep, 'LinC/LinA', ll, r2ll, rmsell, fit_w['rates']['linlin'], w)
            add_row(rows_sweep, 'LinC/NlA',  ln, r2ln, rmseln, fit_w['rates']['linnl'],  w)
            add_row(rows_sweep, 'NlC/NlA',   nn, r2nn, rmsenn, fit_w['rates']['nlnl'],   w)

            sweep_records.append((w, 'LinC/LinA', ll.get('Ecorr', np.nan), ll.get('i0', np.nan) * 1e6))
            sweep_records.append((w, 'LinC/NlA',  ln.get('Ecorr', np.nan), ln.get('i0', np.nan) * 1e6))
            sweep_records.append((w, 'NlC/NlA',   nn.get('Ecorr', np.nan), nn.get('i0', np.nan) * 1e6))

        # ---------------------
        # NEW: Per-file window sensitivity plots
        # ---------------------
        if len(sweep_records) > 0:
            sw_df = pd.DataFrame(sweep_records, columns=['Window (V)', 'Method', 'Ecorr (V)', 'Icorr (µA)'])

            # Icorr vs window
            plt.figure(figsize=(7, 4))
            for meth in sw_df['Method'].unique():
                d = sw_df[sw_df['Method'] == meth].sort_values('Window (V)')
                plt.plot(d['Window (V)'], d['Icorr (µA)'], marker='o', label=meth)
            plt.xlabel('Half-window (V)')
            plt.ylabel('Icorr (µA)')
            plt.title(f"{out_prefix}: Window Sensitivity (Icorr)")
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix}_window_sensitivity_Icorr.png"), dpi=150)
            plt.close()

            # Ecorr vs window
            plt.figure(figsize=(7, 4))
            for meth in sw_df['Method'].unique():
                d = sw_df[sw_df['Method'] == meth].sort_values('Window (V)')
                plt.plot(d['Window (V)'], d['Ecorr (V)'], marker='o', label=meth)
            plt.xlabel('Half-window (V)')
            plt.ylabel('Ecorr (V)')
            plt.title(f"{out_prefix}: Window Sensitivity (Ecorr)")
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix}_window_sensitivity_Ecorr.png"), dpi=150)
            plt.close()

    # ---------------------
    # Outputs
    # ---------------------
    default_df = pd.DataFrame(rows_default)
    if len(default_df) == 0:
        print('Empty summary. No valid CSVs processed.')
        return

    print(default_df.to_string(index=False))
    default_df.to_csv('tafel_summary.csv', index=False)
    print('✅ Default summary saved: tafel_summary.csv')

    sweep_df = pd.DataFrame(rows_sweep)
    if len(sweep_df) > 0:
        sweep_df.to_csv('tafel_window_sensitivity.csv', index=False)
        print('✅ Window sensitivity table saved: tafel_window_sensitivity.csv')

if __name__ == '__main__':
    main()
