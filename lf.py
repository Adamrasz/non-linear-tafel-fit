import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score

# Constants
F = 96485    # C/mol
R = 8.314    # J/mol/K
T = 298      # K

# Surface area and corrosion parameters
A_cm2 = 0.3167  # cm²
EW = 7.021      # g/eq
rho = 2.33      # g/cm³
K = 3272        # mm·g/(C·cm³·year)

# Optimized smoothing function
def optimize_smoothing(I_raw):
    best_score = np.inf
    best_sm = I_raw
    for wl in range(11, 101, 2):
        for po in (2, 3):
            if po >= wl:
                continue
            sm = savgol_filter(I_raw, wl, po)
            rms = np.sqrt(np.mean((I_raw - sm)**2))
            curvature = np.sum(np.abs(np.diff(sm, n=2)))
            score = rms + 1e-3 * curvature
            if score < best_score:
                best_score, best_sm = score, sm
    return best_sm

# Main batch processing
def main():
    summary = []
    for path in glob.glob('*.csv'):
        if 'tafel' in path.lower(): continue
        df = pd.read_csv(path)
        if not {'Step number', 'Working Electrode (V)', 'Current (A)'}.issubset(df.columns):
            continue

        data = df[df['Step number'] == 3]
        if len(data) < 20: continue

        # Prepare directory for plots
        plot_dir = os.path.splitext(path)[0] + '_LinC_LinA_Plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Prepare data
        E = data['Working Electrode (V)'].values
        I_raw = data['Current (A)'].values * 1e3  # Convert to mA
        I_sm = optimize_smoothing(I_raw)
        Ecorr = E[np.argmin(np.abs(I_sm))]
        eta = E - Ecorr
        mask = (eta >= -0.3) & (eta <= 0.3)
        eta = eta[mask]
        I_sm = I_sm[mask]

        mask_c = I_sm < 0
        mask_a = I_sm > 0
        eta_c = eta[mask_c]
        log_I_c = np.log10(-I_sm[mask_c])
        eta_a = eta[mask_a]
        log_I_a = np.log10(I_sm[mask_a])

        # Weighted linear regression for cathodic
        weights_c = 1 / np.abs(log_I_c)
        weights_a = 1 / np.abs(log_I_a)
        slope_c, inter_c, r_c, _, _ = linregress(eta_c, log_I_c)
        beta_c = abs(1 / slope_c) * 1e3
        alpha_c = 2.303 * R * T * 1000 / (F * beta_c)

        # Weighted linear regression for anodic (fixing intercept to icorr)
        i_corr = 10**inter_c
        inter_a = np.log10(i_corr)
        slope_a = np.sum(weights_a * eta_a * (log_I_a - inter_a)) / np.sum(weights_a * eta_a**2)
        beta_a = abs(1 / slope_a) * 1e3
        alpha_a = 2.303 * R * T * 1000 / (F * beta_a)

        # Metrics
        rmse_c = np.sqrt(mean_squared_error(log_I_c, slope_c * eta_c + inter_c))
        r2_c = r2_score(log_I_c, slope_c * eta_c + inter_c)
        rmse_a = np.sqrt(mean_squared_error(log_I_a, slope_a * eta_a + inter_a))
        r2_a = r2_score(log_I_a, slope_a * eta_a + inter_a)
        corrosion_rate = (i_corr * 1e-3) * K * EW / (rho * A_cm2)

        # Save metrics
        summary.append({
            'Filename': path, 'Method': 'LinC/LinA', 'i_corr_uA': i_corr*1e3,
            'beta_c': beta_c, 'beta_a': beta_a, 'alpha_c': alpha_c, 'alpha_a': alpha_a,
            'Corrosion Rate (mmpy)': corrosion_rate,
            'R2_c': r2_c, 'RMSE_c': rmse_c, 'R2_a': r2_a, 'RMSE_a': rmse_a
        })

        # Plot fitting results
        plt.figure(figsize=(6, 4))
        plt.plot(eta_c, log_I_c, 'bo', label='Cathodic Data')
        plt.plot(eta_c, slope_c * eta_c + inter_c, 'b-', label='Cathodic Fit')
        plt.plot(eta_a, log_I_a, 'ro', label='Anodic Data')
        plt.plot(eta_a, slope_a * eta_a + inter_a, 'r-', label='Anodic Fit')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'LinC_LinA_fit.png'))
        plt.close()

        # Decomposition plot
        eta_line = eta
        eta_fit = np.linspace(-0.3, 0.3, 500)
        models_decomp = [
            ('Lin-Lin', i_corr, alpha_a, alpha_c)
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
    summary_df.to_csv('lin_lin_summary.csv', index=False)

if __name__ == '__main__':
    main()
