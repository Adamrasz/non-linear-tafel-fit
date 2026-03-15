import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit, least_squares
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

# Fitting / plotting settings
TAFEL_WINDOW_V = 0.30      # +/- window around initial Ecorr guess (in V)
ETA_EXCLUDE_V = 0.010      # exclude |eta| < this (avoid BV log singularity around eta=0)
ECORR_SEARCH_V = 0.050     # refined-joint Ecorr grid half-span (in V)


# =====================
# Smoothing
# =====================
def smoothness_metric(y: np.ndarray) -> float:
    # 2nd-difference L1 norm (proxy for curvature)
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
    # clip exponent arguments for numerical stability
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

        # ignore data too close to eta=0 (prevents singular behavior/over-weighting)
        mc = np.abs(eta_c) >= ETA_EXCLUDE_V
        ma = np.abs(eta_a) >= ETA_EXCLUDE_V

        r_c = (slope_c * eta_c + log_i0) - logI_c
        r_a = (slope_a * eta_a + log_i0) - logI_a

        r_c = np.where(mc, r_c, 0.0)
        r_a = np.where(ma, r_a, 0.0)

        return np.concatenate([r_c, r_a])

    # initial guesses
    log_i0_0 = float(np.nanmedian(np.concatenate([logI_c, logI_a])))
    p0 = np.array([Ecorr0, log_i0_0, 10.0, -10.0], dtype=float)

    # bounds: Ecorr in window, slopes sign-constrained
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
    coeff = F / (2.303 * R * T)  # log10(exp(alpha*F*eta/RT)) = alpha*coeff*eta

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

    # convert alpha_a back to an anodic Tafel slope for reporting
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
# Refined Joint Model (Model 4) with Ecorr sweep
# =====================
def fit_cathodic_slope_fixed_icorr(eta_c: np.ndarray, log_I_c: np.ndarray, log_icorr: float):
    best = None

    n = len(eta_c)
    if n < 6:
        return None

    for i in range(n - 5):
        for j in range(i + 5, n + 1):
            x = eta_c[i:j]
            y = log_I_c[i:j]

            den = np.sum(x ** 2)
            if den == 0:
                continue

            slope = np.sum(x * (y - log_icorr)) / den
            y_fit = slope * x + log_icorr

            if np.std(y) == 0 or np.std(y_fit) == 0:
                continue
            r = np.corrcoef(y, y_fit)[0, 1]
            r2 = float(r ** 2)

            if r2 <= 0.995:
                continue

            err = float(np.sum((y - y_fit) ** 2))
            length = j - i

            if (best is None) or (length > best['length']) or (length == best['length'] and err < best['err']):
                best = {'slope': float(slope), 'err': err, 'length': length, 'start': i, 'end': j, 'r2': r2}

    return best


def refined_joint_with_ecorr(
    E_c: np.ndarray,
    logI_c: np.ndarray,
    E_a: np.ndarray,
    logI_a: np.ndarray,
    Ecorr_init: float,
    i0_init: float,
    E_bounds: tuple[float, float],
    plot_dir: str,
    out_prefix: str,
):
    """Grid over Ecorr and i0; for each, refit cathodic slope (fixed intercept) + anodic alpha_a (fixed intercept)."""

    Emin, Emax = E_bounds
    if not (np.isfinite(Ecorr_init) and np.isfinite(i0_init) and i0_init > 0):
        return {'Ecorr': np.nan, 'i0': np.nan, 'alpha_a': np.nan, 'alpha_c': np.nan, 'slope_c': np.nan}

    # Ecorr candidates
    Ec_lo = max(Emin, Ecorr_init - ECORR_SEARCH_V)
    Ec_hi = min(Emax, Ecorr_init + ECORR_SEARCH_V)
    Ec_grid = np.linspace(Ec_lo, Ec_hi, 21)

    log_i0_init = np.log10(i0_init)
    log_i0_grid = np.linspace(log_i0_init - 0.5, log_i0_init + 0.5, 25)

    best = None

    for Ec in Ec_grid:
        # keep only points that are safely on their expected side of Ecorr
        mc_side = E_c <= (Ec - ETA_EXCLUDE_V)
        ma_side = E_a >= (Ec + ETA_EXCLUDE_V)

        if mc_side.sum() < 8 or ma_side.sum() < 8:
            continue

        eta_c = E_c[mc_side] - Ec
        eta_a = E_a[ma_side] - Ec
        y_c = logI_c[mc_side]
        y_a = logI_a[ma_side]

        # anodic slope with fixed intercept log_i0: slope minimizes SSE
        den_a = np.sum(eta_a ** 2)
        if den_a == 0:
            continue

        for log_i0 in log_i0_grid:
            fit_c = fit_cathodic_slope_fixed_icorr(eta_c, y_c, log_i0)
            if fit_c is None:
                continue

            slope_c = fit_c['slope']
            err_c = fit_c['err']

            slope_a = float(np.sum(eta_a * (y_a - log_i0)) / den_a)
            y_a_fit = slope_a * eta_a + log_i0
            err_a = float(np.sum((y_a - y_a_fit) ** 2))

            total = err_c + err_a

            # enforce expected signs
            if not (slope_c < 0 and slope_a > 0):
                continue

            if (best is None) or (total < best['total_error']):
                alpha_a = slope_a * 2.303 * R * T / F
                alpha_c = -slope_c * 2.303 * R * T / F
                best = {
                    'total_error': total,
                    'Ecorr': float(Ec),
                    'log_i0': float(log_i0),
                    'i0': float(10 ** log_i0),
                    'slope_c': float(slope_c),
                    'slope_a': float(slope_a),
                    'alpha_a': float(alpha_a),
                    'alpha_c': float(alpha_c),
                }

    if best is None:
        return {'Ecorr': np.nan, 'i0': np.nan, 'alpha_a': np.nan, 'alpha_c': np.nan, 'slope_c': np.nan}

    # Optional plot: refined joint decomposition
    try:
        eta_model = np.linspace(-0.3, 0.3, 500)
        Ia = best['i0'] * np.exp(np.clip(best['alpha_a'] * F * eta_model / (R * T), -50, 50))
        Ic = -best['i0'] * np.exp(-np.clip(best['alpha_c'] * F * eta_model / (R * T), -50, 50))

        plt.figure(figsize=(6, 4))
        plt.plot(eta_model, (Ia + Ic) * 1e3, 'r-', label='I_model')
        plt.plot(eta_model, Ia * 1e3, 'b--', label='Ia')
        plt.plot(eta_model, Ic * 1e3, 'g--', label='Ic')
        plt.axhline(0, color='k', linestyle=':')
        plt.xlabel('η (V)')
        plt.ylabel('Current (mA)')
        plt.title(f'{out_prefix}: Refined Joint Fit')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{out_prefix}_refined_joint.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return best


# =====================
# Main batch processing
# =====================
def main():
    rows = []

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
        # Load and smooth (use A internally)
        # ---------------------
        E = data['Working Electrode (V)'].to_numpy(dtype=float)
        I_raw = data['Current (A)'].to_numpy(dtype=float)
        I_sm = optimize_savgol_smoothing(I_raw)

        # Initial Ecorr guess and fixed E-window
        Ecorr0 = float(E[np.argmin(np.abs(I_sm))])
        mwin = (E >= Ecorr0 - TAFEL_WINDOW_V) & (E <= Ecorr0 + TAFEL_WINDOW_V)
        if mwin.sum() < 20:
            continue

        E_fit = E[mwin]
        I_fit = I_sm[mwin]
        logI_fit = safe_log10_abs(I_fit)

        # branch split by current sign
        mc = I_fit < 0
        ma = I_fit > 0
        if mc.sum() < 8 or ma.sum() < 8:
            continue

        E_c = E_fit[mc]
        E_a = E_fit[ma]
        logI_c = safe_log10_abs(I_fit[mc])
        logI_a = safe_log10_abs(I_fit[ma])

        E_bounds = (float(E_fit.min()), float(E_fit.max()))

        # ---------------------
        # Fit all models with Ecorr included
        # ---------------------
        linlin = fit_lin_lin_ls(E_c, logI_c, E_a, logI_a, Ecorr0, E_bounds)
        linnl = fit_lin_nl_ls(E_c, logI_c, E_a, logI_a, linlin['Ecorr'], E_bounds)
        nlnl = fit_nl_nl_ls(E_fit, logI_fit, linlin['Ecorr'], E_bounds)

        # refined joint (grid) seeded from best available
        Ec_seed = nlnl['Ecorr'] if np.isfinite(nlnl['Ecorr']) else linlin['Ecorr']
        i0_seed = nlnl['i0'] if np.isfinite(nlnl['i0']) else linlin['i0']
        # refj = refined_joint_with_ecorr(E_c, logI_c, E_a, logI_a, Ec_seed, i0_seed, E_bounds, plot_dir, out_prefix)
        refj = {'Ecorr': np.nan, 'i0': np.nan, 'alpha_a': np.nan, 'alpha_c': np.nan, 'beta_a': np.nan, 'beta_c': np.nan}

        # reference Ecorr for overlay plots (keeps one x-axis)
        Ecorr_ref = nlnl['Ecorr'] if np.isfinite(nlnl['Ecorr']) else linlin['Ecorr']

        # ---------------------
        # Corrosion rates (i0 is current; convert to current density by /A_cm2)
        # ---------------------
        def corr_rate(i0_A: float) -> float:
            if not (np.isfinite(i0_A) and i0_A > 0):
                return np.nan
            return float(i0_A * K * EW / (rho * A_cm2))

        rate_linlin = corr_rate(linlin['i0'])
        rate_linnl = corr_rate(linnl['i0'])
        rate_nlnl = corr_rate(nlnl['i0'])
        rate_refj = corr_rate(refj['i0'])

        # ---------------------
        # Metrics (BV model on log10(|I|) in A)
        # ---------------------
        r2_ll, rmse_ll = bv_metrics_on_log_abs(E_fit, I_fit, linlin['Ecorr'], linlin['i0'], linlin['alpha_a'], linlin['alpha_c'])
        r2_ln, rmse_ln = bv_metrics_on_log_abs(E_fit, I_fit, linnl['Ecorr'], linnl['i0'], linnl['alpha_a'], linnl['alpha_c'])
        r2_nn, rmse_nn = bv_metrics_on_log_abs(E_fit, I_fit, nlnl['Ecorr'], nlnl['i0'], nlnl['alpha_a'], nlnl['alpha_c'])
        r2_rj, rmse_rj = bv_metrics_on_log_abs(E_fit, I_fit, refj['Ecorr'], refj['i0'], refj['alpha_a'], refj['alpha_c'])

        # ---------------------
        # Plots
        # ---------------------
        # 1) Decomposition plots (mA), each on its own Ecorr
        eta_fit = np.linspace(-TAFEL_WINDOW_V, TAFEL_WINDOW_V, 500)

        models_decomp = [
            ('Lin-Lin', linlin),
            ('Lin-Nl', linnl),
            ('Nl-Nl', nlnl),
            ('RefinedJoint', refj),
        ]

        for lbl, md in models_decomp:
            Ec_m = md.get('Ecorr', np.nan)
            i0_m = md.get('i0', np.nan)
            aa_m = md.get('alpha_a', np.nan)
            ac_m = md.get('alpha_c', np.nan)

            if not (np.isfinite(Ec_m) and np.isfinite(i0_m) and np.isfinite(aa_m) and np.isfinite(ac_m) and i0_m > 0 and aa_m > 0 and ac_m > 0):
                continue

            eta_line = E - Ec_m

            arg_a = np.clip(aa_m * F * eta_fit / (R * T), -50, 50)
            arg_c = np.clip(ac_m * F * eta_fit / (R * T), -50, 50)
            Ia = i0_m * np.exp(arg_a)
            Ic = -i0_m * np.exp(-arg_c)

            plt.figure(figsize=(6, 4))
            plt.plot(eta_line, I_sm * 1e3, 'k-', label='Smoothed LSV')
            plt.plot(eta_fit, (Ia + Ic) * 1e3, 'r-', label='I_model')
            plt.plot(eta_fit, Ia * 1e3, 'b--', label='Ia')
            plt.plot(eta_fit, Ic * 1e3, 'g--', label='Ic')
            plt.axhline(0, color='gray', linestyle=':')
            plt.xlabel('η (V)')
            plt.ylabel('Current (mA)')
            plt.title(f'{path}: {lbl} Decomposition')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{out_prefix}_{lbl}_decomp.png"), dpi=150)
            plt.close()

        # 2) Linear baselines on Tafel axes using Lin-Lin fit (log10(mA))
        eta_data_ll = E_fit - linlin['Ecorr']
        logI_mA = logI_fit + 3
        sort_idx = np.argsort(eta_data_ll)

        plt.figure(figsize=(6, 4))
        plt.plot(eta_data_ll[sort_idx], logI_mA[sort_idx], 'k-', label='Data')

        log_i0_ll = linlin['log_i0']
        slope_a_ll = linlin['slope_a']
        slope_c_ll = linlin['slope_c']

        # baseline ranges to eta=0
        x_c = np.linspace(float(np.min(eta_data_ll)), 0.0, 200)
        x_a = np.linspace(0.0, float(np.max(eta_data_ll)), 200)
        plt.plot(x_c, slope_c_ll * x_c + log_i0_ll + 3, 'b--', label=f'Cath Lin (β={linlin["beta_c"]:.1f} mV/dec)')
        plt.plot(x_a, slope_a_ll * x_a + log_i0_ll + 3, 'r--', label=f'Anod Lin (β={linlin["beta_a"]:.1f} mV/dec)')

        plt.axvline(0, linestyle='--', color='gray')
        plt.axhline(log_i0_ll + 3, linestyle='--', color='gray')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|) (mA)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{out_prefix}_tafel_linear_baselines.png"), dpi=150)
        plt.close()

        # 3) Residuals for Lin-Lin piecewise linear (log domain, A)
        # (residuals for display: predicted - data)
        pred_c = slope_c_ll * (E_c - linlin['Ecorr']) + log_i0_ll
        pred_a = slope_a_ll * (E_a - linlin['Ecorr']) + log_i0_ll

        plt.figure(figsize=(6, 4))
        plt.plot(E_c - linlin['Ecorr'], pred_c - logI_c, 'b-', label='Cath Resid')
        plt.plot(E_a - linlin['Ecorr'], pred_a - logI_a, 'r-', label='Anod Resid')
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel('η (V)')
        plt.ylabel('Residual (log10(A))')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{out_prefix}_tafel_residuals.png"), dpi=150)
        plt.close()

        # 4) Nonlinear BV fits (log10(mA)) overlay: Lin-Nl, Nl-Nl, RefinedJoint
        eta_ref = E_fit - Ecorr_ref
        sort_ref = np.argsort(eta_ref)

        plt.figure(figsize=(6, 4))
        plt.plot(eta_ref[sort_ref], (logI_fit + 3)[sort_ref], 'k-', label='Data')

        E_grid = np.linspace(float(E_fit.min()), float(E_fit.max()), 400)
        eta_grid_ref = E_grid - Ecorr_ref

        for lbl, md in [('Lin-Nl', linnl), ('Nl-Nl', nlnl), ('Refined', refj)]:
            Ec_m = md.get('Ecorr', np.nan)
            i0_m = md.get('i0', np.nan)
            aa_m = md.get('alpha_a', np.nan)
            ac_m = md.get('alpha_c', np.nan)
            if not (np.isfinite(Ec_m) and np.isfinite(i0_m) and np.isfinite(aa_m) and np.isfinite(ac_m) and i0_m > 0 and aa_m > 0 and ac_m > 0):
                continue
            y = log_bv(E_grid - Ec_m, i0_m, aa_m, ac_m) + 3
            plt.plot(eta_grid_ref, y, label=lbl)

        plt.axvline(0, linestyle='--', color='gray')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|) (mA)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{out_prefix}_tafel_nonlinear_tafel.png"), dpi=150)
        plt.close()

        # 5) Combined BV model comparisons on Tafel axes (log10(mA))
        plt.figure(figsize=(6, 4))
        plt.plot(eta_ref[sort_ref], (logI_fit + 3)[sort_ref], 'k-', label='Data')

        for lbl, md in [('Lin-Lin', linlin), ('Lin-Nl', linnl), ('Nl-Nl', nlnl), ('Refined', refj)]:
            Ec_m = md.get('Ecorr', np.nan)
            i0_m = md.get('i0', np.nan)
            aa_m = md.get('alpha_a', np.nan)
            ac_m = md.get('alpha_c', np.nan)
            if not (np.isfinite(Ec_m) and np.isfinite(i0_m) and np.isfinite(aa_m) and np.isfinite(ac_m) and i0_m > 0 and aa_m > 0 and ac_m > 0):
                continue
            y = log_bv(E_grid - Ec_m, i0_m, aa_m, ac_m) + 3
            plt.plot(eta_grid_ref, y, label=lbl)

        plt.axvline(0, linestyle='--', color='gray')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|) (mA)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{out_prefix}_tafel_combined_models.png"), dpi=150)
        plt.close()

        # ---------------------
        # Summary rows (per-file, per-method)
        # ---------------------
        def add_row(method: str, md: dict, r2v: float, rmsev: float, ratev: float):
            rows.append({
                'Filename': path,
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

        add_row('LinC/LinA', linlin, r2_ll, rmse_ll, rate_linlin)
        add_row('LinC/NlA', linnl, r2_ln, rmse_ln, rate_linnl)
        add_row('NlC/NlA', nlnl, r2_nn, rmse_nn, rate_nlnl)
        add_row('RefinedJoint', refj, r2_rj, rmse_rj, rate_refj)

    # ---------------------
    # Summary table
    # ---------------------
    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        print('Empty summary. No valid CSVs processed.')
        return

    print(summary_df.to_string(index=False))
    summary_df.to_csv('tafel_summary.csv', index=False)
    print('✅ Summary table generated and saved: tafel_summary.csv')


if __name__ == '__main__':
    main()
