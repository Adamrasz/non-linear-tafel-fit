import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
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

# Butler-Volmer current model
def bv_current(eta, i0, alpha_a, alpha_c):
    arg_a = np.clip(alpha_a * F * eta / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta / (R * T), -50, 50)
    return i0 * (np.exp(arg_a) - np.exp(-arg_c))

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
        plot_dir = os.path.splitext(path)[0] + '_NlC_NlA_Plots'
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

        # Full nonlinear fitting with weighting
        eta_full = eta
        log_I_full = np.log10(np.abs(I_sm) + 1e-20)
        weights = 1 / np.abs(log_I_full)
        weights[np.isinf(weights)] = 1

        # Dynamic initial guesses and parameter bounds
        i0_guess = np.median(np.abs(I_sm))
        initial_params = [i0_guess, 0.5, 0.5]

        # Grid search and curve fit
        best_fit = {'error': np.inf}
        for log_i0 in np.linspace(np.log10(i0_guess)-1, np.log10(i0_guess)+1, 5):
            for alpha_a in np.linspace(0.1, 0.9, 5):
                for alpha_c in np.linspace(0.1, 0.9, 5):
                    try:
                        popt, _ = curve_fit(
                            lambda eta, i0, aa, ac: np.log10(np.abs(bv_current(eta, i0, aa, ac)) + 1e-20),
                            eta_full, log_I_full, sigma=weights, absolute_sigma=False,
                            p0=[10**log_i0, alpha_a, alpha_c], bounds=(0, [np.inf, 1, 1]))
                        fit_log = np.log10(np.abs(bv_current(eta_full, *popt)) + 1e-20)
                        error = np.sum(weights * (log_I_full - fit_log)**2)
                        if error < best_fit['error']:
                            best_fit = {'params': popt, 'error': error}
                    except:
                        continue

        i0_f, alpha_a_f, alpha_c_f = best_fit['params']

        # Metrics
        fit_I = bv_current(eta_full, i0_f, alpha_a_f, alpha_c_f)
        r2 = r2_score(log_I_full, np.log10(np.abs(fit_I) + 1e-20))
        rmse = np.sqrt(mean_squared_error(log_I_full, np.log10(np.abs(fit_I) + 1e-20)))
        corrosion_rate = (i0_f * 1e-3) * K * EW / (rho * A_cm2)

        # Save metrics
        summary.append({
            'Filename': path, 'Method': 'NlC/NlA', 'i0_uA': i0_f*1e3,
            'alpha_a': alpha_a_f, 'alpha_c': alpha_c_f, 'Corrosion Rate (mmpy)': corrosion_rate,
            'R2': r2, 'RMSE': rmse
        })

        # Plot fitting result
        plt.figure(figsize=(6, 4))
        plt.plot(eta_full, log_I_full, 'k-', label='LSV Data')
        plt.plot(eta_full, np.log10(np.abs(fit_I)+1e-20), 'r--', label='NlC/NlA Fit')
        plt.xlabel('η (V)')
        plt.ylabel('log10(|I|)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'NlC_NlA_fit.png'))
        plt.close()

        # Decomposition plot
        eta_line = eta
        eta_fit = np.linspace(-0.3, 0.3, 500)
        models_decomp = [
            ('Nl-Nl', i0_f, alpha_a_f, alpha_c_f)
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
    summary_df.to_csv('nlc_nla_summary.csv', index=False)

if __name__ == '__main__':
    main()
