import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
from itertools import product
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
    best = {'score': np.inf}
    for wl in range(11, 101, 2):
        for po in [2,3]:
            if po >= wl: continue
            try:
                I_sm = savgol_filter(I_raw, wl, po)
            except:
                continue
            rms = np.sqrt(np.mean((I_raw - I_sm)**2))
            curv = smoothness_metric(I_sm)
            score = rms + 1e-3 * curv
            if score < best['score']:
                best.update({'score': score, 'I_sm': I_sm})
    return best['I_sm']

# BV current model
def bv_current(eta, i0, alpha_a, alpha_c):
    arg_a = np.clip(alpha_a * F * eta / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta / (R * T), -50, 50)
    return i0 * (np.exp(arg_a) - np.exp(-arg_c))

# Cathodic slope fitting helper
" + 
"def fit_cathodic_slope_fixed_icorr(eta_c, log_I_c, log_icorr):
" +
"    best = {'length': 0}
" +
"    for i in range(len(eta_c) - 5):
" +
"        for j in range(i + 5, len(eta_c)):
" +
"            x, y = eta_c[i:j], log_I_c[i:j]
" +
"            slope = np.sum(x * (y - log_icorr)) / np.sum(x**2)
" +
"            y_fit = slope * x + log_icorr
" +
"            r2 = np.corrcoef(y, y_fit)[0, 1]**2
" +
"            if r2 > 0.995 and (j - i) > best['length']:
" +
"                err = np.sum((y - y_fit)**2)
" +
"                best.update({'slope': slope, 'err': err, 'length': j - i})
" +
"    return best if 'slope' in best else None

" +
"# Refined joint model (Model 4)
def final_refined_joint_model(eta_c, log_I_c, eta_a, log_I_a, icorr_init):
" +
"    # Joint Tafel-BV fit: cathodic slope + anodic BV
" +
"    log_range = np.linspace(np.log10(icorr_init)-0.5, np.log10(icorr_init)+0.5, 25)
" +
"    best = {'total_error': np.inf}
" +
"    for log_ic in log_range:
" +
"        fit_c = fit_cathodic_slope_fixed_icorr(eta_c, log_I_c, log_ic)
" +
"        if not fit_c:
" +
"            continue
" +
"        slope_c, err_c = fit_c['slope'], fit_c['err']
" +
"        icorr = 10**log_ic
" +
"        def log_bv_a(e, aa):
" +
"            arg = np.clip(aa * F * e / (R * T), -50, 50)
" +
"            return np.log10(icorr * np.exp(arg) + 1e-20)
" +
"        try:
" +
"            alpha_a, _ = curve_fit(log_bv_a, eta_a, log_I_a, p0=[0.3], bounds=(0,1))
" +
"            err_a = np.sum((log_I_a - log_bv_a(eta_a, alpha_a[0]))**2)
" +
"        except:
" +
"            continue
" +
"        total = err_c + err_a
" +
"        if total < best['total_error']:
" +
"            best.update({'total_error': total, 'i0': icorr, 'alpha_a': alpha_a[0], 'slope_c': slope_c})
" +
"    i0 = best.get('i0', np.nan)
" +
"    slope_c = best.get('slope_c', 1)
" +
"    alpha_a = best.get('alpha_a', 0.5)
" +
"    alpha_c = 2.303 * R * T / (F * ((1 / slope_c) / 1000))
" +
"    return {'i0': i0, 'alpha_a': alpha_a, 'alpha_c': alpha_c}


# Refined joint model (Model 4)
# (plots use the main loop's plot_dir)

    log_range = np.linspace(np.log10(icorr_init)-0.5, np.log10(icorr_init)+0.5, 25)
    best = {'total_error': np.inf}
    for log_ic in log_range:
        fit_c = fit_cathodic_slope_fixed_icorr(eta_c, log_I_c, log_ic)
        if not fit_c:
            continue
        slope_c, err_c = fit_c['slope'], fit_c['err']
        icorr = 10**log_ic
        def log_bv_a(eta, alpha_a):
            arg = np.clip(alpha_a * F * eta / (R * T), -50, 50)
            return np.log10(icorr * np.exp(arg) + 1e-20)
        try:
            popt, _ = curve_fit(log_bv_a, eta_a, log_I_a, p0=[0.3], bounds=(0,1))
            alpha_a = popt[0]
            err_a = np.sum((log_I_a - log_bv_a(eta_a, alpha_a))**2)
        except:
            continue
        total = err_c + err_a
        if total < best['total_error']:
            best.update({'total_error': total, 'icorr': icorr, 'slope_c': slope_c, 'alpha_a': alpha_a})
    i0 = best.get('icorr', np.nan)
    slope_c = best.get('slope_c', 1)
    alpha_a = best.get('alpha_a', 0.5)
    alpha_c = 2.303 * R * T / (F * ((1 / slope_c) / 1000))
    eta_model = np.linspace(-0.3, 0.3, 500)
    arg_a = np.clip(alpha_a * F * eta_model / (R * T), -50, 50)
    arg_c = np.clip(alpha_c * F * eta_model / (R * T), -50, 50)
    Ia = i0 * np.exp(arg_a)
    Ic = -i0 * np.exp(-arg_c)
    plt.figure(figsize=(8,5))
    plt.plot(eta_model, Ia+Ic, 'r-', label='I_model')
    plt.plot(eta_model, Ia, 'b--', label='Ia')
    plt.plot(eta_model, Ic, 'g--', label='Ic')
    plt.axhline(0, color='k', linestyle=':')
    plt.xlabel('η (V)'); plt.ylabel('Current (mA)')
    plt.title(f'{out_prefix}: Refined Joint Fit')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{out_prefix}_refined_joint.png"))
    plt.close()

    return {'i0': i0, 'alpha_a': alpha_a, 'alpha_c': alpha_c}

# Batch processing
summary = []
for path in glob.glob('*.csv'):
    # skip summary or other non-LSV CSVs
    if 'tafel' in path.lower():
        continue
    # create a single folder for this file's plots
    plot_dir = os.path.splitext(path)[0] + '_plots'
    os.makedirs(plot_dir, exist_ok=True)
    # skip summary or other non-LSV CSVs
    if 'tafel' in path.lower():
        continue
    # create a single folder for this file's plots
    plot_dir = os.path.splitext(path)[0] + '_plots'
    os.makedirs(plot_dir, exist_ok=True)
    try:
        df = pd.read_csv(path)
    except:
        continue
    if not {'Step number','Working Electrode (V)','Current (A)'}.issubset(df.columns): continue
    data = df[df['Step number']==3]
    if len(data) < 20: continue

    # Prepare data
    E = np.array(data['Working Electrode (V)'])
    I_raw = np.array(data['Current (A)']) * 1e3  # mA
    I_sm = optimize_savgol_smoothing(I_raw)
    Ecorr = E[np.argmin(np.abs(I_sm))]
    eta = E - Ecorr
    mask = (eta >= -0.3) & (eta <= 0.3)
    eta_t = eta[mask]; I_t = I_sm[mask]
    mask_c = I_t < 0; mask_a = I_t > 0
    if mask_c.sum() < 6 or mask_a.sum() < 6: continue
    eta_c, log_I_c = eta_t[mask_c], np.log10(-I_t[mask_c])
    eta_a, log_I_a = eta_t[mask_a], np.log10(I_t[mask_a])

    # Linear Tafel fits
    best = {'len': 0}
    for i in range(len(eta_c)-5):
        for j in range(i+5, len(eta_c)):
            s, inter, r, *_ = linregress(eta_c[i:j], log_I_c[i:j])
            if r**2 > 0.995 and (j-i) > best['len']:
                best.update({'slope': s, 'inter': inter, 'len': j-i, 'r2': r**2})
    slope_c, inter_c = best['slope'], best['inter']
    i_corr = 10**inter_c
    beta_c = abs(1/slope_c)*1e3  # mV/dec
    alpha_c_lin = 2.303 * R * T * 1000 / (F * beta_c)  # convert beta_c mV/dec to V/dec for alpha calculation
    r2_c = best['r2']
    rmse_c = np.sqrt(mean_squared_error(log_I_c, slope_c*eta_c + inter_c))
    corrosion_rate = i_corr * K * EW / (rho * A_cm2)

    inter_a = np.log10(i_corr)
    slope_a = np.sum(eta_a * (log_I_a - inter_a)) / np.sum(eta_a**2)
    beta_a = abs(1/slope_a)*1e3
    alpha_a_lin = 2.303 * R * T * 1000 / (F * beta_a)  # convert beta_a mV/dec to V/dec for alpha calculation
    r2_a = r2_score(log_I_a, slope_a*eta_a + inter_a)
    rmse_a = np.sqrt(mean_squared_error(log_I_a, slope_a*eta_a + inter_a))

    # Nonlinear Tafel fits
    popt_c, _ = curve_fit(lambda e, ac: np.log10(i_corr*np.exp(-ac*F*e/(R*T))+1e-20), eta_c, log_I_c, p0=[0.3], bounds=(0,1))
    alpha_c_nl = popt_c[0]
    popt_a, _ = curve_fit(lambda e, aa: np.log10(i_corr*np.exp(aa*F*e/(R*T))+1e-20), eta_a, log_I_a, p0=[0.3], bounds=(0,1))
    alpha_a_nl = popt_a[0]
    # compute log fits
    log_fit_c_nl = np.log10(i_corr*np.exp(-alpha_c_nl*F*eta_c/(R*T)) + 1e-20)
    log_fit_a_nl = np.log10(i_corr*np.exp(alpha_a_nl*F*eta_a/(R*T)) + 1e-20)

    # Full BV
    eta_full = np.concatenate([eta_c, eta_a])
    I_full = np.concatenate([-10**log_I_c, 10**log_I_a])
    try:
        popt_f, _ = curve_fit(lambda e,i0,aa,ac: bv_current(e,i0,aa,ac), eta_full, I_full,
                              p0=[i_corr, alpha_a_nl, alpha_c_nl], bounds=(0,[np.inf,1,1]))
        i0_f, alpha_a_f, alpha_c_f = popt_f
    except:
        i0_f, alpha_a_f, alpha_c_f = np.nan, alpha_a_lin, alpha_c_lin

    # Refined joint
    joint = final_refined_joint_model(eta_c, log_I_c, eta_a, log_I_a, i_corr, path)
    i0_j, alpha_a_j, alpha_c_j = joint['i0'], joint['alpha_a'], joint['alpha_c']

    # Plot outputs
    # Decomposition plots
    eta_line = eta
    eta_fit = np.linspace(-0.3, 0.3, 500)
    models_decomp = [
        ('Lin-Lin', i_corr, alpha_a_lin, alpha_c_lin),
        ('Lin-Nl', i_corr, alpha_a_nl, alpha_c_lin),
        ('Nl-Nl', i0_f, alpha_a_f, alpha_c_f),
        ('RefinedJoint', i0_j, alpha_a_j, alpha_c_j)
    ]
    for lbl, i0v, aav, acv in models_decomp:
        plt.figure(figsize=(6,4))
        plt.plot(eta_line, I_sm, 'k-', label='Smoothed LSV')
        arg_a = np.clip(aav*F*eta_fit/(R*T), -50, 50)
        arg_c = np.clip(acv*F*eta_fit/(R*T), -50, 50)
        Ia = i0v*np.exp(arg_a)
        Ic = -i0v*np.exp(-arg_c)
        plt.plot(eta_fit, Ia+Ic, 'r-', label='I_model')
        plt.plot(eta_fit, Ia, 'b--', label='Ia')
        plt.plot(eta_fit, Ic, 'g--', label='Ic')
        plt.axhline(0, color='gray', linestyle=':')
        plt.xlabel('η (V)'); plt.ylabel('Current (mA)')
        plt.title(f'{path}: {lbl} Decomposition')
        plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(path)[0]}_{lbl}_decomp.png"))
        plt.close()

    # Linear baselines
    plt.figure(figsize=(6,4))
    plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
    plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
    x_c = np.linspace(min(eta_c), 0, 100)
    plt.plot(x_c, slope_c*x_c+inter_c, 'b--', label=f'Cathodic Lin (β={beta_c:.1f})')
    x_a = np.linspace(0, max(eta_a), 100)
    plt.plot(x_a, slope_a*x_a+inter_a, 'r--', label=f'Anodic Lin (β={beta_a:.1f})')
    plt.xlabel('η (V)'); plt.ylabel('log10(|I|)')
    plt.grid(True); plt.legend(loc='best'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(path)[0]}_tafel_linear_baselines.png"))
    plt.close()

    # Residuals
    plt.figure(figsize=(6,4))
    plt.plot(eta_c, slope_c*eta_c+inter_c-log_I_c, 'b-')
    plt.plot(eta_a, slope_a*eta_a+inter_a-log_I_a, 'r-')
    plt.xlabel('η (V)'); plt.ylabel('Residual')
    plt.legend(['Cathodic','Anodic'], loc='best'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(path)[0]}_tafel_residuals.png"))
    plt.close()

    # Nonlinear Tafel
    plt.figure(figsize=(6,4))
    plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
    plt.plot(eta_c, log_fit_c_nl, 'b--', label=f'Cathodic NL (α={alpha_c_nl:.2f})')
    plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
    plt.plot(eta_a, log_fit_a_nl, 'r--', label=f'Anodic NL (α={alpha_a_nl:.2f})')
    plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.grid(True)
    plt.legend(loc='best'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(path)[0]}_tafel_nonlinear_tafel.png"))
    plt.close()

        # Combined Tafel model comparisons
    plt.figure(figsize=(6,4))
    plt.plot(eta_c, log_I_c, 'k-', label='Cathodic Data')
    plt.plot(eta_a, log_I_a, 'k-', label='Anodic Data')
    models_combined = [
        ('Lin-Lin', i_corr, alpha_a_lin, alpha_c_lin),
        ('Lin-Nl', i_corr, alpha_a_nl, alpha_c_lin),
        ('Nl-Nl', i0_f, alpha_a_f, alpha_c_f),
        ('Refined', i0_j, alpha_a_j, alpha_c_j)
    ]
    eta_plot = np.linspace(min(eta), max(eta), 200)
    for lbl, i0v, aav, acv in models_combined:
        if lbl == 'Lin-Lin':
            # Linear both branches
            log_model = np.where(eta_plot < 0,
                                 slope_c * eta_plot + inter_c,
                                 slope_a * eta_plot + inter_a)
        elif lbl == 'Lin-Nl':
            # Linear cathodic, nonlinear anodic
            log_model = np.where(eta_plot < 0,
                                 slope_c * eta_plot + inter_c,
                                 np.log10(i_corr * np.exp(alpha_a_nl * F * eta_plot / (R * T)) + 1e-20))
        else:
            # Nonlinear or refined full BV
            log_model = np.log10(np.abs(bv_current(eta_plot, i0v, aav, acv)) + 1e-20)
        plt.plot(eta_plot, log_model, label=lbl)
    plt.xlabel('η (V)')
    plt.ylabel('log10(|I|)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(path)[0]}_tafel_combined_models.png"))
    plt.close()

        # Summary row: collect metrics for each method
    # Compute additional metrics for nonlinear and refined joint
    # Full-BV fit metrics
    # r2_full and rmse_full already computed? assume we have r2_full_nl and rmse_full_nl

    summary.append({
        'Filename': path,
        'beta_c': beta_c,
        'beta_a_lin': beta_a,
        'alpha_c_lin': alpha_c_lin,
        'alpha_a_lin': alpha_a_lin,
        'r2_c': r2_c,
        'rmse_c': rmse_c,
        'r2_a': r2_a,
        'rmse_a': rmse_a,
        'alpha_c_nl': alpha_c_nl,
        'alpha_a_nl': alpha_a_nl,
        'r2_c_nl': r2_c_nl if 'r2_c_nl' in locals() else np.nan,
        'rmse_c_nl': rmse_c_nl if 'rmse_c_nl' in locals() else np.nan,
        'r2_a_nl': r2_a_nl if 'r2_a_nl' in locals() else np.nan,
        'rmse_a_nl': rmse_a_nl if 'rmse_a_nl' in locals() else np.nan,
        'i_corr_uA': i_corr * 1e3,
        'rate_lin_lin': corrosion_rate,
        'i_corr_nl_uA': i0_f * 1e3,
        'rate_nl_nl': i0_f * K * EW / (rho * A_cm2),
        'i0_j_uA': i0_j * 1e3,
        'rate_joint': i0_j * K * EW / (rho * A_cm2),
        'alpha_c_j': alpha_c_j,
        'alpha_a_j': alpha_a_j,
        'r2_joint': r2_joint if 'r2_joint' in locals() else np.nan,
        'rmse_joint': rmse_joint if 'rmse_joint' in locals() else np.nan
    })

# === Build and display summary table ===
# Prepare Method summary per file (aggregated across models)
methods = ['LinC/LinA', 'LinC/NlA', 'NlC/NlA', 'RefinedJoint']
# For each file, expand metrics into multi-row summary
rows = []
for entry in summary:
    fn = entry['Filename']
    # Linear-linear
    rows.append({
        'Filename': fn,
        'Method': 'LinC/LinA',
        'βc (mV/dec)': entry.get('beta_c', np.nan),
        'βa (mV/dec)': entry.get('beta_a_lin', np.nan),
        'αc': entry.get('alpha_c_lin', np.nan),
        'αa': entry.get('alpha_a_lin', np.nan),
        'Icorr (µA)': entry['i_corr_uA'],
        'Rate (mmpy)': entry['rate_lin_lin'],
        'R²_lin': entry.get('r2_c', np.nan),
        'RMSE_lin': entry.get('rmse_c', np.nan),
        'R²_nl': entry.get('r2_c_nl', np.nan),
        'RMSE_nl': entry.get('rmse_c_nl', np.nan)
    })
    # Linear-nonlinear
    rows.append({
        'Filename': fn,
        'Method': 'LinC/NlA',
        'βc (mV/dec)': entry.get('beta_c', np.nan),
        'βa (mV/dec)': entry.get('beta_a_nl', np.nan),
        'αc': entry.get('alpha_c_lin', np.nan),
        'αa': entry.get('alpha_a_nl', np.nan),
        'Icorr (µA)': entry['i_corr_uA'],
        'Rate (mmpy)': entry['rate_lin_lin'],
        'R²_lin': entry.get('r2_a', np.nan),
        'RMSE_lin': entry.get('rmse_a', np.nan),
        'R²_nl': entry.get('r2_a_nl', np.nan),
        'RMSE_nl': entry.get('rmse_a_nl', np.nan)
    })
    # Nonlinear-nonlinear
    rows.append({
        'Filename': fn,
        'Method': 'NlC/NlA',
        'βc (mV/dec)': entry.get('beta_c_nl', np.nan),
        'βa (mV/dec)': entry.get('beta_a_nl', np.nan),
        'αc': entry.get('alpha_c_nl', np.nan),
        'αa': entry.get('alpha_a_nl', np.nan),
        'Icorr (µA)': entry.get('i_corr_nl_uA', np.nan),
        'Rate (mmpy)': entry.get('rate_nl_nl', np.nan),
        'R²_nl': entry.get('r2_full_nl', np.nan),
        'RMSE_nl': entry.get('rmse_full_nl', np.nan)
    })
    # Refined joint
    rows.append({
        'Filename': fn,
        'Method': 'RefinedJoint',
        'βc (mV/dec)': entry.get('beta_c_j', np.nan),
        'βa (mV/dec)': entry.get('beta_a_j', np.nan),
        'αc': entry.get('alpha_c_j', np.nan),
        'αa': entry.get('alpha_a_j', np.nan),
        'Icorr (µA)': entry.get('i0_j_uA', np.nan),
        'Rate (mmpy)': entry.get('rate_joint', np.nan),
        'R²_joint': entry.get('r2_joint', np.nan),
        'RMSE_joint': entry.get('rmse_joint', np.nan)
    })
# Create DataFrame
summary_df = pd.DataFrame(rows)
# Print summary table
print(summary_df.to_string(index=False))
# Save to CSV
summary_df.to_csv('tafel_summary.csv', index=False)
print('✅ Summary table generated and saved.')
