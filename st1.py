import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# Constants
F = 96485    # C/mol
R = 8.314    # J/mol/K
T = 298      # K
# Surface area and corrosion conversion
A_cm2 = 0.3167   # cm^2
EW = 7.021       # g/eq
rho = 2.33       # g/cm^3
K = 3272         # mm·g/(C·cm3·year)

# Smoothing helper
def smoothness_metric(y):
    return np.sum(np.abs(np.diff(y, n=2)))

def optimize_savgol_smoothing(I_raw):
    best_score = np.inf
    best_sm = I_raw
    for wl in range(11, 101, 2):
        for po in (2,3):
            if po >= wl:
                continue
            try:
                sm = savgol_filter(I_raw, wl, po)
            except:
                continue
            score = np.sqrt(np.mean((I_raw - sm)**2)) + 1e-3 * smoothness_metric(sm)
            if score < best_score:
                best_score, best_sm = score, sm
    return best_sm

# BV current model
def bv_current(eta, i0, alpha_a, alpha_c):
    arg_a = np.clip(alpha_a * F * eta / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta / (R * T), -50, 50)
    return i0 * (np.exp(arg_a) - np.exp(-arg_c))

# Cathodic slope fitting helper
def fit_cathodic_slope_fixed_icorr(eta_c, log_I_c, log_icorr):
    best = {'length': 0}
    for i in range(len(eta_c) - 5):
        for j in range(i + 5, len(eta_c)):
            x = eta_c[i:j]
            y = log_I_c[i:j]
            slope = np.sum(x * (y - log_icorr)) / np.sum(x**2)
            y_fit = slope * x + log_icorr
            r2 = np.corrcoef(y, y_fit)[0, 1]**2
            if r2 > 0.995 and (j - i) > best['length']:
                err = np.sum((y - y_fit)**2)
                best = {'slope': slope, 'err': err, 'length': j - i}
    return best if 'slope' in best else None

# Refined joint model
def final_refined_joint_model(eta_c, log_I_c, eta_a, log_I_a, icorr_init):
    best = {'total_error': np.inf}
    for log_ic in np.linspace(np.log10(icorr_init) - 0.5, np.log10(icorr_init) + 0.5, 25):
        fit_c = fit_cathodic_slope_fixed_icorr(eta_c, log_I_c, log_ic)
        if not fit_c:
            continue
        i0 = 10**log_ic
        slope_c = fit_c['slope']
        err_c = fit_c['err']
        def log_bv_a(e, aa):
            arg = np.clip(aa * F * e / (R * T), -50, 50)
            return np.log10(i0 * np.exp(arg) + 1e-20)
        try:
            popt, _ = curve_fit(log_bv_a, eta_a, log_I_a, p0=[0.3], bounds=(0,1))
            alpha_a = popt[0]
            err_a = np.sum((log_I_a - log_bv_a(eta_a, alpha_a))**2)
        except:
            continue
        total = err_c + err_a
        if total < best['total_error']:
            best = {'i0': i0, 'alpha_a': alpha_a, 'slope_c': slope_c, 'total_error': total}
    i0 = best.get('i0', np.nan)
    slope_c = best.get('slope_c', 1)
    alpha_a = best.get('alpha_a', 0.5)
    alpha_c = 2.303 * R * T * 1000 / (F * (abs(1 / slope_c) * 1000))
    return {'i0': i0, 'alpha_a': alpha_a, 'alpha_c': alpha_c}

# Main batch processing
def main():
    summary = []
    for path in glob.glob('*.csv'):
        # skip summary files
        if 'tafel' in path.lower():
            continue
        # read and validate file
        try:
            df = pd.read_csv(path)
        except:
            continue
        if not {'Step number', 'Working Electrode (V)', 'Current (A)'}.issubset(df.columns):
            continue
        data = df[df['Step number'] == 3]
        if len(data) < 20:
            continue
        # create plot directory
        plot_dir = os.path.splitext(path)[0] + '_plots'
        os.makedirs(plot_dir, exist_ok=True)
        # Prepare data
        E = data['Working Electrode (V)'].values
        I_raw = data['Current (A)'].values * 1e3
        I_sm = optimize_savgol_smoothing(I_raw)
        Ecorr = E[np.argmin(np.abs(I_sm))]
        eta = E - Ecorr
        mask = (eta >= -0.3) & (eta <= 0.3)
        eta_t = eta[mask]
        I_t = I_sm[mask]
        mask_c = I_t < 0
        mask_a = I_t > 0
        if mask_c.sum() < 6 or mask_a.sum() < 6:
            continue
        eta_c = eta_t[mask_c]
        log_I_c = np.log10(-I_t[mask_c])
        eta_a = eta_t[mask_a]
        log_I_a = np.log10(I_t[mask_a])
        # Linear Tafel fits
        best_lin = {'length': 0}
        for i in range(len(eta_c) - 5):
            for j in range(i + 5, len(eta_c)):
                s, inter, r, *_ = linregress(eta_c[i:j], log_I_c[i:j])
                if r**2 > 0.995 and (j - i) > best_lin['length']:
                    best_lin = {'slope': s, 'inter': inter, 'length': j - i, 'r2': r**2}
        slope_c = best_lin['slope']
        inter_c = best_lin['inter']
        r2_c = best_lin['r2']
        i_corr = 10**inter_c
        beta_c = abs(1 / slope_c) * 1e3
        alpha_c_lin = 2.303 * R * T * 1000 / (F * beta_c)
        rmse_c = np.sqrt(mean_squared_error(log_I_c, slope_c * eta_c + inter_c))
        rate_lin_lin = (i_corr * 1e-3) * K * EW / (rho * A_cm2)
        # Anodic linear fit
        inter_a = np.log10(i_corr)
        slope_a = np.sum(eta_a * (log_I_a - inter_a)) / np.sum(eta_a**2)
        beta_a = abs(1 / slope_a) * 1e3
        alpha_a_lin = 2.303 * R * T * 1000 / (F * beta_a)
        r2_a = r2_score(log_I_a, slope_a * eta_a + inter_a)
        rmse_a = np.sqrt(mean_squared_error(log_I_a, slope_a * eta_a + inter_a))
        # Prepare weights based on log current magnitudes
        weights = 1 / np.abs(np.concatenate([log_I_c, log_I_a]))
        weights[np.isinf(weights)] = 1  # Handle division by zero safely

        # Log-domain BV current function for weighted fitting
        def log_bv_current(eta, i0, alpha_a, alpha_c):
            return np.log10(np.abs(bv_current(eta, i0, alpha_a, alpha_c)) + 1e-20)

        # Weighted nonlinear full BV fit
        eta_full = np.concatenate([eta_c, eta_a])
        log_I_full = np.concatenate([log_I_c, log_I_a])
        try:
            popt_f, _ = curve_fit(log_bv_current, eta_full, log_I_full,
                                  sigma=weights, absolute_sigma=False,
                                  p0=[i_corr, alpha_a_nl, alpha_c_nl],
                                  bounds=([0, 0, 0], [np.inf, 1, 1]))
            i0_f, alpha_a_f, alpha_c_f = popt_f
        except:
            i0_f, alpha_a_f, alpha_c_f = np.nan, alpha_a_lin, alpha_c_lin
            i0_f, alpha_a_f, alpha_c_f = np.nan, alpha_a_lin, alpha_c_lin
                # Prepare raw domain for full BV χ² metrics
        eta_full = np.concatenate([eta_c, eta_a])
        I_full = np.concatenate([-10**log_I_c, 10**log_I_a])
        # Compute chi-squared metrics for full BV
        res_full = I_full - bv_current(eta_full, i0_f, alpha_a_f, alpha_c_f)
        chi2_full = np.sum(res_full**2)
        chi2_norm_full = np.sum((res_full / I_full)**2)

        # Refined joint model

        eta_full = np.concatenate([eta_c, eta_a])
        I_full = np.concatenate([-10**log_I_c, 10**log_I_a])
        try:
            popt_f, _ = curve_fit(lambda e, i0, aa, ac: bv_current(e, i0, aa, ac), eta_full, I_full, p0=[i_corr, alpha_a_nl, alpha_c_nl], bounds=(0,[np.inf,1,1]))
            i0_f, alpha_a_f, alpha_c_f = popt_f
        except:
            i0_f, alpha_a_f, alpha_c_f = np.nan, alpha_a_lin, alpha_c_lin
        # Refined joint model
        joint = final_refined_joint_model(eta_c, log_I_c, eta_a, log_I_a, i_corr)
        i0_j = joint['i0']
        alpha_a_j = joint['alpha_a']
        alpha_c_j = joint['alpha_c']
        # Plotting: decomposition, baselines, residuals, etc.
        # Each call to plt.savefig uses os.path.join(plot_dir, ...)
        # [Plotting code omitted for brevity]
                        # Compute additional metrics for summary
        # Full-BV fit metrics (using finite-mask)
        full_model_log = np.log10(np.abs(bv_current(eta_full, i0_f, alpha_a_f, alpha_c_f)) + 1e-20)
        data_log_full = np.concatenate([log_I_c, log_I_a])
        mask_full = np.isfinite(full_model_log) & np.isfinite(data_log_full)
        r2_full_nl = r2_score(data_log_full[mask_full], full_model_log[mask_full])
        rmse_full_nl = np.sqrt(mean_squared_error(data_log_full[mask_full], full_model_log[mask_full]))

        # Cathodic nonlinear metrics
        mask_c_nl = np.isfinite(log_fit_c_nl) & np.isfinite(log_I_c)
        r2_c_nl = r2_score(log_I_c[mask_c_nl], log_fit_c_nl[mask_c_nl])
        rmse_c_nl = np.sqrt(mean_squared_error(log_I_c[mask_c_nl], log_fit_c_nl[mask_c_nl]))

        # Anodic nonlinear metrics
        mask_a_nl = np.isfinite(log_fit_a_nl) & np.isfinite(log_I_a)
        r2_a_nl = r2_score(log_I_a[mask_a_nl], log_fit_a_nl[mask_a_nl])
        rmse_a_nl = np.sqrt(mean_squared_error(log_I_a[mask_a_nl], log_fit_a_nl[mask_a_nl]))
        rate_nl_nl = (i0_f * 1e-3) * K * EW / (rho * A_cm2)
        rate_joint = (i0_j * 1e-3) * K * EW / (rho * A_cm2)

        # --- Plotting Section ---
        # Decomposition plots
        eta_line = eta
        eta_fit = np.linspace(-0.3, 0.3, 500)
        models_decomp = [
            ('Lin-Lin', i_corr, alpha_a_lin, alpha_c_lin),
            ('Lin-Nl', i_corr, alpha_a_nl, alpha_c_lin),
            ('Nl-Nl', i0_f, alpha_a_f, alpha_c_f),
            ('Refined', i0_j, alpha_a_j, alpha_c_j)
        ]
        for lbl, i0v, aav, acv in models_decomp:
            plt.figure(figsize=(6,4))
            plt.plot(eta_line, I_sm, 'k-', label='Smoothed LSV')
            arg_a = np.clip(aav * F * eta_fit / (R * T), -50, 50)
            arg_c = np.clip(acv * F * eta_fit / (R * T), -50, 50)
            Ia = i0v * np.exp(arg_a)
            Ic = -i0v * np.exp(-arg_c)
            plt.plot(eta_fit, Ia+Ic, 'r-', label='I_model')
            plt.plot(eta_fit, Ia, 'b--', label='Ia')
            plt.plot(eta_fit, Ic, 'g--', label='Ic')
            plt.axhline(0, color='gray', linestyle=':')
            plt.xlabel('η (V)'); plt.ylabel('Current (mA)'); plt.title(f'{path}: {lbl}')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{lbl}_decomp.png'))
            plt.close()

        # Linear baselines
        plt.figure(figsize=(6,4))
        plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
        plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
        x_c = np.linspace(min(eta_c), 0, 100)
        plt.plot(x_c, slope_c * x_c + inter_c, 'b--', label=f'Cathodic Lin (β={beta_c:.1f})')
        x_a = np.linspace(0, max(eta_a), 100)
        plt.plot(x_a, slope_a * x_a + inter_a, 'r--', label=f'Anodic Lin (β={beta_a:.1f})')
        plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'tafel_linear_baselines.png'))
        plt.close()

        # Residuals
        plt.figure(figsize=(6,4))
        plt.plot(eta_c, slope_c * eta_c + inter_c - log_I_c, 'b-', label='Cathodic Resid')
        plt.plot(eta_a, slope_a * eta_a + inter_a - log_I_a, 'r-', label='Anodic Resid')
        plt.xlabel('η (V)'); plt.ylabel('Residual'); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'tafel_residuals.png'))
        plt.close()

        # Nonlinear Tafel fits
        plt.figure(figsize=(6,4))
        plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
        plt.plot(eta_c, log_fit_c_nl, 'b--', label=f'Cathodic NL (α={alpha_c_nl:.2f})')
        plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
        plt.plot(eta_a, log_fit_a_nl, 'r--', label=f'Anodic NL (α={alpha_a_nl:.2f})')
        plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'tafel_nonlinear.png'))
        plt.close()

        # Combined Tafel models
        plt.figure(figsize=(6,4))
        plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
        plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
        eta_plot = np.linspace(min(eta), max(eta), 200)
        for lbl2, i0v2, aav2, acv2 in models_decomp:
            if lbl2 == 'Lin-Lin':
                log_model = np.where(eta_plot < 0, slope_c * eta_plot + inter_c, slope_a * eta_plot + inter_a)
            elif lbl2 == 'Lin-Nl':
                log_model = np.where(eta_plot < 0, slope_c * eta_plot + inter_c, np.log10(i_corr * np.exp(alpha_a_nl * F * eta_plot / (R * T)) + 1e-20))
            else:
                log_model = np.log10(np.abs(bv_current(eta_plot, i0v2, aav2, acv2)) + 1e-20)
            plt.plot(eta_plot, log_model, label=lbl2)
        plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'tafel_combined.png'))
        plt.close()

        # Refined joint plot
        eta_m = np.linspace(-0.3, 0.3, 500)
        Ia = i0_j * np.exp(alpha_a_j * F * eta_m / (R * T))
        Ic = -i0_j * np.exp(-alpha_c_j * F * eta_m / (R * T))
        plt.figure(figsize=(6,4))
        plt.plot(eta_m, Ia + Ic, 'r-')
        plt.plot(eta_m, Ia, 'b--')
        plt.plot(eta_m, Ic, 'g--')
        plt.axhline(0, color='gray', linestyle=':')
        plt.xlabel('η (V)'); plt.ylabel('Current (mA)'); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'refined_joint.png'))
        plt.close()

                        # --- Compute χ² and normalized χ² for each model (log domain) ---
        # Linear-linear (LinC/LinA)
        res_lin_c_log = slope_c * eta_c + inter_c - log_I_c
        res_lin_a_log = slope_a * eta_a + inter_a - log_I_a
        chi2_lin = np.sum(res_lin_c_log**2 + res_lin_a_log**2)
        chi2_norm_lin = np.sum((res_lin_c_log/log_I_c)**2 + (res_lin_a_log/log_I_a)**2)
        # Linear-Nonlinear (LinC/NlA)
        res_nl_a_log = log_fit_a_nl - log_I_a
        chi2_lin_nl = np.sum(res_lin_c_log**2 + res_nl_a_log**2)
        chi2_norm_lin_nl = np.sum((res_lin_c_log/log_I_c)**2 + (res_nl_a_log/log_I_a)**2)
        # Nonlinear-Nonlinear (NlC/NlA)
        log_full_model = np.log10(np.abs(bv_current(eta_full, i0_f, alpha_a_f, alpha_c_f)) + 1e-20)
        data_log_full = np.concatenate([log_I_c, log_I_a])
        res_full_log = log_full_model - data_log_full
        chi2_nl_nl = np.sum(res_full_log**2)
        chi2_norm_nl_nl = np.sum((res_full_log/data_log_full)**2)
        # Refined Joint
        log_joint_model = np.log10(np.abs(bv_current(eta_full, i0_j, alpha_a_j, alpha_c_j)) + 1e-20)
        res_joint_log = log_joint_model - data_log_full
        chi2_j = np.sum(res_joint_log**2)
        chi2_norm_j = np.sum((res_joint_log/data_log_full)**2)

                # Append all metrics for each model
        # LinC/LinA: linear cathodic & linear anodic fits
        summary.append({
            'Filename': path,
            'Method': 'LinC/LinA',
            'i0_uA': i_corr * 1e3,
            'beta_c': beta_c,
            'beta_a': beta_a,
            'alpha_c': alpha_c_lin,
            'alpha_a': alpha_a_lin,
            'Rate': rate_lin_lin,
            'R2': r2_c,
            'RMSE': rmse_c,
            'Chi2': chi2_lin,
            'Chi2_norm': chi2_norm_lin
        })
        # LinC/NlA: linear cathodic, nonlinear anodic fits
        summary.append({
            'Filename': path,
            'Method': 'LinC/NlA',
            'i0_uA': i_corr * 1e3,
            'beta_c': beta_c,
            'beta_a': (2.303 * R * T * 1000) / (F * alpha_a_nl) if not np.isnan(alpha_a_nl) else np.nan,
            'alpha_c': alpha_c_lin,
            'alpha_a': alpha_a_nl,
            'Rate': rate_lin_lin,
            'R2': r2_a,
            'RMSE': rmse_a,
            'Chi2': chi2_lin_nl,
            'Chi2_norm': chi2_norm_lin_nl
        })
        # NlC/NlA: nonlinear both branches (full BV)
        summary.append({
            'Filename': path,
            'Method': 'NlC/NlA',
            'i0_uA': i0_f * 1e3,
            'beta_c': (2.303 * R * T * 1000) / (F * alpha_c_f) if not np.isnan(alpha_c_f) else np.nan,
            'beta_a': (2.303 * R * T * 1000) / (F * alpha_a_f) if not np.isnan(alpha_a_f) else np.nan,
            'alpha_c': alpha_c_f,
            'alpha_a': alpha_a_f,
            'Rate': rate_nl_nl,
            'R2': r2_full_nl,
            'RMSE': rmse_full_nl,
            'Chi2': chi2_nl_nl,
            'Chi2_norm': chi2_norm_nl_nl
        })
        # RefinedJoint: refined joint BV fits
        summary.append({
            'Filename': path,
            'Method': 'RefinedJoint',
            'i0_uA': i0_j * 1e3,
            'beta_c': (2.303 * R * T * 1000) / (F * alpha_c_j) if not np.isnan(alpha_c_j) else np.nan,
            'beta_a': (2.303 * R * T * 1000) / (F * alpha_a_j) if not np.isnan(alpha_a_j) else np.nan,
            'alpha_c': alpha_c_j,
            'alpha_a': alpha_a_j,
            'Rate': rate_joint,
            'R2': r2_joint,
            'RMSE': rmse_joint,
            'Chi2': chi2_j,
            'Chi2_norm': chi2_norm_j
        })
    # End loop
    # Build summary DataFrame from collected metrics
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('tafel_summary.csv', index=False)

if __name__ == '__main__':
    main()
