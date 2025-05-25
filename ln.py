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
F = 96485
R = 8.314
T = 298
A_cm2 = 0.3167
EW = 7.021
rho = 2.33
K = 3272

# Optimized smoothing function
def optimize_smoothing(I_raw):
    best_score, best_sm = np.inf, I_raw
    for wl in range(11, 101, 2):
        for po in (2, 3):
            if po >= wl:
                continue
            sm = savgol_filter(I_raw, wl, po)
            score = np.sqrt(np.mean((I_raw - sm)**2)) + 1e-3 * np.sum(np.abs(np.diff(sm, n=2)))
            if score < best_score:
                best_score, best_sm = score, sm
    return best_sm

# Anodic BV model fixed at log_icorr
def log_anodic_bv_fixed_icorr(eta, alpha_a, log_icorr):
    return log_icorr + (alpha_a * F * eta) / (2.303 * R * T)

# BV current model
def bv_current(eta, i0, alpha_a, alpha_c):
    arg_a = np.clip(alpha_a * F * eta / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta / (R * T), -50, 50)
    return i0 * (np.exp(arg_a) - np.exp(-arg_c))

# Main processing
def main():
    summary = []
    for path in glob.glob('*.csv'):
        if 'tafel' in path.lower(): continue
        df = pd.read_csv(path)
        if not {'Step number', 'Working Electrode (V)', 'Current (A)'}.issubset(df.columns): continue

        data = df[df['Step number'] == 3]
        if len(data) < 20: continue

        plot_dir = os.path.splitext(path)[0] + '_Optimized_LinC_NlA'
        os.makedirs(plot_dir, exist_ok=True)

        E = data['Working Electrode (V)'].values
        I_sm = optimize_smoothing(data['Current (A)'].values * 1e3)
        Ecorr = E[np.argmin(np.abs(I_sm))]
        eta = E - Ecorr
        mask = (eta >= -0.3) & (eta <= 0.3)
        eta, I_sm = eta[mask], I_sm[mask]

        mask_c, mask_a = I_sm < 0, I_sm > 0
        eta_c, log_I_c = eta[mask_c], np.log10(-I_sm[mask_c])
        eta_a, log_I_a = eta[mask_a], np.log10(I_sm[mask_a])

        # Dynamic linear cathodic optimization
        best_lin = {'error': np.inf}
        for i in range(len(eta_c)-5):
            for j in range(i+5, len(eta_c)):
                s, ic, r, _, _ = linregress(eta_c[i:j], log_I_c[i:j])
                err = np.sum((log_I_c[i:j] - (s*eta_c[i:j]+ic))**2)
                if err < best_lin['error']:
                    best_lin = {'slope': s, 'icorr': ic, 'error': err}

        slope_c, log_icorr = best_lin['slope'], best_lin['icorr']
        beta_c = abs(1/slope_c)*1e3
        alpha_c = 2.303*R*T*1000/(F*beta_c)
        i_corr = 10**log_icorr

        # Nonlinear anodic fit constrained at log_icorr
        weights_a = 1 / np.abs(log_I_a)
        alpha_a, _ = curve_fit(lambda eta, alpha_a: log_anodic_bv_fixed_icorr(eta, alpha_a, log_icorr),
                               eta_a, log_I_a, sigma=weights_a, bounds=(0.05, 1), p0=[0.5])
        alpha_a = alpha_a[0]
        beta_a = 2.303*R*T*1000/(F*alpha_a)

        # Metrics
        fit_log_a = log_anodic_bv_fixed_icorr(eta_a, alpha_a, log_icorr)
        fit_log_c = slope_c*eta_c + log_icorr
        rmse = np.sqrt(mean_squared_error(np.concatenate([log_I_c, log_I_a]),
                                          np.concatenate([fit_log_c, fit_log_a])))
        r2 = r2_score(np.concatenate([log_I_c, log_I_a]), np.concatenate([fit_log_c, fit_log_a]))
        corrosion_rate = (i_corr*1e-3)*K*EW/(rho*A_cm2)

        summary.append({
            'Filename': path, 'Method': 'Optimized LinC/NlA', 'i_corr_uA': i_corr*1e3,
            'beta_c': beta_c, 'beta_a': beta_a, 'alpha_c': alpha_c, 'alpha_a': alpha_a,
            'Corrosion Rate (mmpy)': corrosion_rate, 'R2': r2, 'RMSE': rmse
        })

        # Combined experimental data
        eta_comb = np.concatenate([eta_c, eta_a])
        log_I_comb = np.concatenate([log_I_c, log_I_a])
        sort_idx = np.argsort(eta_comb)
        eta_comb, log_I_comb = eta_comb[sort_idx], log_I_comb[sort_idx]

        # Plot 1: Optimized LinC/NlA Fit
        plt.figure(figsize=(6,4))
        plt.plot(eta_comb, log_I_comb, 'k-', label='Experimental Data')
        plt.plot(eta_c, fit_log_c, 'b--', label='Cathodic Linear Fit')
        plt.plot(eta_a, fit_log_a, 'r--', label='Anodic Nonlinear Fit')
        plt.axvline(0, color='gray', linestyle=':')
        plt.xlabel('η (V)'); plt.ylabel('log10(|I|)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Optimized_LinC_NlA_fit.png'))
        plt.close()

        # Plot 2: Log BV overlay
        eta_plot = np.linspace(min(eta), max(eta), 500)
      #  log_model = np.where(eta_plot < 0,
       #                      slope_c * eta_plot + log_icorr,
       #                      log_anodic_bv_fixed_icorr(eta_plot, alpha_a, log_icorr))
       # plt.figure(figsize=(6,4))
      #  plt.plot(eta_comb, log_I_comb, 'k-', label='Experimental Data')
      #  plt.plot(eta_plot, log_model, 'm-', label='Combined Log BV Model')
      #  plt.axvline(0, color='gray', linestyle=':')
      #  plt.xlabel('η (V)'); plt.ylabel('log10(|I|)')
       # plt.legend(); plt.tight_layout()
      #  plt.savefig(os.path.join(plot_dir, 'Log_BV_Overlay.png'))
       # plt.close()

        # Plot fitting result
        plt.figure(figsize=(6, 4))
        plt.plot(eta_comb, log_I_comb, 'k-', label='LSV Data')
        log_model = np.log10(np.abs(bv_current(eta_plot, i_corr, alpha_a, alpha_c)) + 1e-20)
        plt.plot(eta_plot, log_model, label='LinC/NlA Fit')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Log_LinC_NlA_fit.png'))
        plt.close()

         # Decomposition plot
        eta_line = eta
        eta_fit = np.linspace(-0.3, 0.3, 500)
        models_decomp = [
            ('Lin-Nl', i_corr, alpha_a, alpha_c)
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
            plt.savefig(os.path.join(plot_dir, 'decomp.png'))
            plt.close()

    summary_df = pd.DataFrame(summary)
    print(summary_df)
    summary_df.to_csv('optimized_linC_nlA_summary.csv', index=False)

if __name__ == '__main__':
    main()
