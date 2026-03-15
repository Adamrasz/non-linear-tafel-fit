import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle

# Constants
F = 96485    # C/mol
R = 8.314    # J/mol/K
T = 298      # K
# Surface area and corrosion parameters
a_cm2 = 0.3167  # cm²
EW = 7.021      # g/eq
rho = 2.33      # g/cm³
K = 3272        # mm·g/(C·cm³·year)

# Smoothing function
def optimize_smoothing(I_raw):
    best_score = np.inf
    best_sm = I_raw
    for wl in range(11, 101, 2):
        for po in (2, 3):
            if po >= wl:
                continue
            sm = savgol_filter(I_raw, wl, po)
            rms = np.sqrt(np.mean((I_raw - sm) ** 2))
            curvature = np.sum(np.abs(np.diff(sm, n=2)))
            score = rms + 1e-3 * curvature
            if score < best_score:
                best_score, best_sm = score, sm
    return best_sm

# BV residual function for least_squares (Tafel axes)
def bv_residuals(params, E, logI):
    Ecorr, log_i0, alpha_a, alpha_c = params
    i0 = 10 ** log_i0
    eta = E - Ecorr
    I_pred = i0 * (
        np.exp(alpha_a * F * eta / (R * T))
        - np.exp(-alpha_c * F * eta / (R * T))
    )
    return np.log10(np.abs(I_pred) + 1e-20) - logI

# Main batch processing
# Walks through all subfolders of a parent data directory,
# processes each CSV found, and writes outputs back to its folder.
def main(data_parent: str):
    data_parent = os.path.abspath(os.path.expanduser(data_parent))

    if not os.path.exists(data_parent):
        raise FileNotFoundError(f"DATA_PARENT does not exist: {data_parent}")

    summary = []
    found_any_csv = False
    processed_any = False

    print(f"Scanning for CSV files under: {data_parent}")

    # Traverse all subdirectories
    for root, _, files in os.walk(data_parent):
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue
            if 'tafel' in fname.lower():
                continue

            found_any_csv = True
            csv_path = os.path.join(root, fname)

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[SKIP] Failed to read: {csv_path} ({e})")
                continue

            # Validate required columns
            required = {'Step number', 'Working Electrode (V)', 'Current (A)'}
            if not required.issubset(df.columns):
                print(f"[SKIP] Missing required columns in: {csv_path}")
                continue

            # only process Step number 3 scans
            data = df[df['Step number'] == 3]
            if len(data) < 20:
                print(f"[SKIP] Too few Step=3 points (<20) in: {csv_path}")
                continue

            processed_any = True
            print(f"[RUN ] {csv_path}")

            # Prepare output folder in same directory
            plot_dir = os.path.join(root, 'BV_Fit')
            os.makedirs(plot_dir, exist_ok=True)

            # Load and smooth
            E = data['Working Electrode (V)'].values
            I_raw = data['Current (A)'].values
            I_sm = optimize_smoothing(I_raw)

            # Initial Ecorr guess & Tafel window
            Ecorr0 = E[np.argmin(np.abs(I_sm))]
            mask = (E >= Ecorr0 - 0.3) & (E <= Ecorr0 + 0.3)
            E_fit = E[mask]
            I_fit = I_sm[mask]

            if len(E_fit) < 20:
                print(f"[SKIP] Too few points within ±300 mV of Ecorr0 in: {csv_path}")
                continue

            I_fit = np.where(I_fit == 0, 1e-20, I_fit)
            logI_fit = np.log10(np.abs(I_fit))

            # Nonlinear least-squares fit
            p0 = [Ecorr0, np.mean(logI_fit) - 1, 0.5, 0.5]
            bounds = ([E_fit.min(), -np.inf, 0.01, 0.01], [E_fit.max(), np.inf, 1, 1])
            res = least_squares(
                bv_residuals,
                p0,
                args=(E_fit, logI_fit),
                bounds=bounds,
                xtol=1e-12,
            )

            Ecorr_opt, log_i0_opt, alpha_a, alpha_c = res.x
            i0_opt = 10 ** log_i0_opt

            beta_a = 2.303 * R * T / (alpha_a * F) * 1000
            beta_c = 2.303 * R * T / (alpha_c * F) * 1000

            resid = res.fun
            chi2 = float(np.sum(resid ** 2))
            N = len(resid)
            chi_sqrtN = chi2 / np.sqrt(N)
            chi2_I2 = float(np.sum(resid ** 2 / (logI_fit ** 2 + 1e-12)))

            eta = E_fit - Ecorr_opt
            I_model = i0_opt * (
                np.exp(alpha_a * F * eta / (R * T))
                - np.exp(-alpha_c * F * eta / (R * T))
            )

            # 1) BV Tafel Fit Plot
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.plot(eta, logI_fit + 3, 'k.', label='Data')
            ax.plot(eta, np.log10(np.abs(I_model)) + 3, 'r-', label='BV Fit')
            ax.axvline(0, ls='--', c='gray')
            ax.axhline(log_i0_opt + 3, ls='--', c='gray')
            ax.set_xlabel('η (V)')
            ax.set_ylabel('log10(|I|) (mA)')
            ax.set_title(fname)

            leg = ax.legend(loc='lower left', frameon=True)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1)

            txt = AnchoredText(
                f"Ecorr={Ecorr_opt:.3f} V\nIcorr={i0_opt * 1e6:.2f} µA",
                loc='lower right',
                prop=dict(size=9),
                pad=0.3,
                frameon=True,
            )
            txt.patch.set_edgecolor('black')
            txt.patch.set_linewidth(1)
            ax.add_artist(txt)

            ax2 = ax.twinx()
            # Use the correct scale for the density axis (log10(mA/cm^2))
            ax2.plot(eta, np.log10(np.abs(I_fit) * 1e3 / a_cm2), '', alpha=0)
            ax2.set_ylabel('log10 Current Density (mA/cm²)')

            ax.grid(True)
            fig.savefig(os.path.join(plot_dir, 'BV_tafel_fit.png'), dpi=150)
            plt.close(fig)

            # 2) Linear Baseline Plot
            slope_a_lin = 1000.0 / beta_a
            slope_c_lin = -1000.0 / beta_c

            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.plot(eta, logI_fit + 3, 'k-', label='Data')
            ax.plot(eta, slope_c_lin * eta + log_i0_opt + 3, 'b--', label=f'Cath Baseline βc={beta_c:.1f}')
            ax.plot(eta, slope_a_lin * eta + log_i0_opt + 3, 'r--', label=f'Anod Baseline βa={beta_a:.1f}')
            ax.axvline(0, ls='--', c='gray')
            ax.axhline(log_i0_opt + 3, ls='--', c='gray')
            ax.set_xlabel('η (V)')
            ax.set_ylabel('log10(|I|) (mA)')
            ax.set_title(fname + ' Linear Baselines')

            leg = ax.legend(loc='lower left', frameon=True)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1)

            txt = AnchoredText(
                f"Ecorr={Ecorr_opt:.3f} V\nIcorr={i0_opt * 1e6:.2f} µA",
                loc='lower right',
                prop=dict(size=9),
                pad=0.3,
                frameon=True,
            )
            txt.patch.set_edgecolor('black')
            txt.patch.set_linewidth(1)
            ax.add_artist(txt)

            ax2 = ax.twinx()
            ax2.plot(eta, np.log10(np.abs(I_fit) * 1e3 / a_cm2), '', alpha=0)
            ax2.set_ylabel('log10 Current Density (mA/cm²)')

            ax.grid(True)
            fig.savefig(os.path.join(plot_dir, 'BV_linear_baselines.png'), dpi=150)
            plt.close(fig)

            # 3) Decomposition Plot with Anchored Inset
            Ia = i0_opt * np.exp(alpha_a * F * eta / (R * T))
            Ic = -i0_opt * np.exp(-alpha_c * F * eta / (R * T))

            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.plot(eta, I_fit * 1e3, 'k-', label='LSV Data')
            ax.plot(eta, Ia * 1e3, 'r--', label='Anodic I')
            ax.plot(eta, Ic * 1e3, 'b--', label='Cathodic I')
            ax.axhline(0, ls='--', c='gray')
            ax.set_xlabel('η (V)')
            ax.set_ylabel('Current (mA)')
            ax.set_title(fname + ' BV Decomposition')

            leg = ax.legend(loc='upper left', frameon=True)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1)

            ax.grid(True)

            ax2 = ax.twinx()
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim(ymin / a_cm2, ymax / a_cm2)
            ax2.set_ylabel('Current Density (mA/cm²)')

            # Draw white bg box in axes coords
            bg_box = [0.60, 0.08, 0.35, 0.35]
            rect = Rectangle(
                (bg_box[0], bg_box[1]),
                bg_box[2],
                bg_box[3],
                transform=ax.transAxes,
                facecolor='white',
                edgecolor='black',
                linewidth=0.8,
                zorder=5,
            )
            ax.add_patch(rect)

            # Inset anchored inside that box
            in_box = [0.62, 0.10, 0.30, 0.30]
            axins = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=in_box,
                bbox_transform=ax.transAxes,
                loc='lower left',
            )

            mask_zoom = (eta >= -0.1) & (eta <= 0.1)
            axins.plot(eta[mask_zoom], I_fit[mask_zoom] * 1e3, 'k-')
            axins.plot(eta[mask_zoom], Ia[mask_zoom] * 1e3, 'r--')
            axins.plot(eta[mask_zoom], Ic[mask_zoom] * 1e3, 'b--')
            axins.set_xlim(-0.1, 0.1)

            if np.any(mask_zoom):
                ylo = float(np.min(np.concatenate([Ic[mask_zoom], I_fit[mask_zoom], Ia[mask_zoom]]) * 1e3))
                yhi = float(np.max(np.concatenate([Ic[mask_zoom], I_fit[mask_zoom], Ia[mask_zoom]]) * 1e3))
                if ylo == yhi:
                    ylo -= 1
                    yhi += 1
                axins.set_ylim(ylo, yhi)

            axins.grid(True)
            axins.tick_params(labelsize=6)

            fig.savefig(os.path.join(plot_dir, 'BV_decomposition.png'), dpi=150)
            plt.close(fig)

            # Record summary
            corr_rate = i0_opt * K * EW / (rho * a_cm2)
            summary.append(
                {
                    'Filename': csv_path,
                    'Ecorr (V)': Ecorr_opt,
                    'Icorr (µA)': i0_opt * 1e6,
                    'beta_a (mV/dec)': beta_a,
                    'beta_c (mV/dec)': beta_c,
                    'chi2': chi2,
                    'chi2/sqrtN': chi_sqrtN,
                    'chi2/I2': chi2_I2,
                    'Rate (mmpy)': corr_rate,
                }
            )

    if not found_any_csv:
        print("No CSV files found under the selected DATA_PARENT.")

    if found_any_csv and not processed_any:
        print("CSV files were found, but none matched the expected format (Step number == 3 and required columns).")

    # Final summary CSV in parent folder
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(data_parent, 'bv_fit_summary.csv'), index=False)
    print(df_sum)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch BV fit on all experiment CSVs under a folder tree."
    )
    parser.add_argument(
        '--data_parent',
        default=os.getcwd(),
        help='Top-level folder to scan recursively for experiment CSVs (default: current working directory).',
    )

    args = parser.parse_args()
    main(args.data_parent)
