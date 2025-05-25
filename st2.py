# Stern Fit Script: Complete Automated Analysis + Plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# --- Constants ---
F = 96485        # C/mol
R = 8.314        # J/mol/K
T = 298          # K
A_cm2 = 0.3167   # cm^2
EW = 7.021       # g/eq
rho = 2.33       # g/cm^3
K = 3272         # mmpy conversion constant

# --- Load & Smooth Data ---
df = pd.read_csv("LSV.csv")
E = df["Working Electrode (V)"].values
I_raw = df["Current (mA)"].values / 1000  # Convert to A
I = savgol_filter(I_raw, window_length=11, polyorder=3)

# --- Estimate Ecorr & Icorr ---
idx_zero = np.argmin(np.abs(I))
Ecorr = E[idx_zero]
# Icorr will come from linear cathodic intercept

# --- Define Tafel Region ---
mask_tafel = (E >= Ecorr - 0.3) & (E <= Ecorr + 0.3)
E_t = E[mask_tafel]; I_t = I[mask_tafel]

# --- Split Branches ---
mask_c = I_t < 0; mask_a = I_t > 0
E_c = E_t[mask_c]; I_c = -I_t[mask_c]  # positive for log
E_a = E_t[mask_a]; I_a = I_t[mask_a]
eta_c = E_c - Ecorr; log_I_c = np.log10(I_c)
eta_a = E_a - Ecorr; log_I_a = np.log10(I_a)

# --- Derivative Plot for Cathodic ---
d1 = np.gradient(log_I_c, eta_c)
d2 = np.gradient(d1, eta_c)
plt.figure()
plt.plot(eta_c, d1, label='1st deriv')
plt.plot(eta_c, d2, label='2nd deriv')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Overpotential η (V)'); plt.ylabel('Derivative')
plt.title('Cathodic Derivatives')
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('cathodic_derivative.png')

# --- Automated Cathodic Linear Fit ---
best_c = {'r2': 0, 'length': 0}
for i in range(len(eta_c) - 5):
    for j in range(i + 5, len(eta_c)):
        x = eta_c[i:j]; y = log_I_c[i:j]
        slope, intercept, r, *_ = linregress(x, y)
        err = np.mean(np.abs(y - (slope * x + intercept)))
        if r**2 > 0.995 and err < 0.05 and (j - i) > best_c['length']:
            best_c.update({'start': i, 'end': j, 'slope': slope, 'intercept': intercept,
                           'r2': r**2, 'mse': mean_squared_error(y, slope * x + intercept), 'length': j - i})
start_c, end_c = best_c['start'], best_c['end']
slope_c, intercept_c = best_c['slope'], best_c['intercept']
beta_c = abs(1 / slope_c) * 1000  # mV/dec
r2_c = best_c['r2']; mse_c = best_c['mse']; rmse_c = np.sqrt(mse_c)
# Compute residuals for cathodic linear fit
residuals_c = log_I_c - (slope_c * eta_c + intercept_c)
i_corr = 10 ** intercept_c

# --- Automated Anodic Linear Fit (forced through Icorr) ---
# Anchor intercept at log10(Icorr)
intercept_a_lin = np.log10(i_corr)
# Perform least-squares fit for slope with fixed intercept
# slope minimizes sum((slope*eta_a + intercept - log_I_a)^2)
slope_a_lin = np.sum(eta_a * (log_I_a - intercept_a_lin)) / np.sum(eta_a**2)
# Compute metrics
y_fit_a_lin = slope_a_lin * eta_a + intercept_a_lin
residuals_a = log_I_a - y_fit_a_lin
r2_a = r2_score(log_I_a, y_fit_a_lin)
mse_a = mean_squared_error(log_I_a, y_fit_a_lin)
rmse_a = np.sqrt(mse_a)
# Tafel slope
beta_a_lin = abs(1 / slope_a_lin) * 1000  # mV/dec

# --- Nonlinear BV Fit on Tafel Domain ---
def log_bv(eta, i0, alpha_a, alpha_c):
    i = i0 * (np.exp(alpha_a * F * eta / (R * T)) - np.exp(-alpha_c * F * eta / (R * T)))
    # avoid log(0) by adding a small constant
    return np.log10(np.abs(i) + 1e-20)
# Cathodic nonlinear (alpha_c)
popt_c, _ = curve_fit(lambda e, ac: log_bv(e, i_corr, 0, ac), eta_c, log_I_c, p0=[0.3])
alpha_c_nl = popt_c[0]; beta_c_nl = 2.303 * R * T / (alpha_c_nl * F) * 1000
y_c_nl = log_bv(eta_c, i_corr, 0, alpha_c_nl)
mask_c_nl = np.isfinite(y_c_nl) & np.isfinite(log_I_c)
r2_c_nl = r2_score(log_I_c[mask_c_nl], y_c_nl[mask_c_nl])
rmse_c_nl = np.sqrt(mean_squared_error(log_I_c[mask_c_nl], y_c_nl[mask_c_nl]))
# Anodic nonlinear (alpha_a)
popt_a, _ = curve_fit(lambda e, aa: log_bv(e, i_corr, aa, 0), eta_a, log_I_a, p0=[0.3])
alpha_a_nl = popt_a[0]; beta_a_nl = 2.303 * R * T / (alpha_a_nl * F) * 1000
y_a_nl = log_bv(eta_a, i_corr, alpha_a_nl, 0)
mask_a_nl = np.isfinite(y_a_nl) & np.isfinite(log_I_a)
r2_a_nl = r2_score(log_I_a[mask_a_nl], y_a_nl[mask_a_nl])
rmse_a_nl = np.sqrt(mean_squared_error(log_I_a[mask_a_nl], y_a_nl[mask_a_nl]))
# Full nonlinear fit (fit i0, alpha_a, alpha_c)
# Fit i0 plus transfer coefficients to entire Tafel data
i0_guess = i_corr
popt_f, _ = curve_fit(
    lambda e, i0, aa, ac: log_bv(e, i0, aa, ac),
    np.concatenate([eta_c, eta_a]),
    np.concatenate([log_I_c, log_I_a]),
    p0=[i0_guess, 0.3, 0.3],
    bounds=(0, [np.inf, 1, 1])
)
i0_nl, alpha_a_f, alpha_c_f = popt_f
# Compute derived slopes
beta_a_f = 2.303 * R * T / (alpha_a_f * F) * 1000
beta_c_f = 2.303 * R * T / (alpha_c_f * F) * 1000
# New Tafel log-model
y_full_nl = log_bv(np.concatenate([eta_c, eta_a]), i0_nl, alpha_a_f, alpha_c_f)
mask_full_nl = np.isfinite(y_full_nl) & np.isfinite(np.concatenate([log_I_c, log_I_a]))
# Fit metrics
r2_full_nl = r2_score(np.concatenate([log_I_c, log_I_a])[mask_full_nl], y_full_nl[mask_full_nl])
rmse_full_nl = np.sqrt(mean_squared_error(np.concatenate([log_I_c, log_I_a])[mask_full_nl], y_full_nl[mask_full_nl]))
# Use new Icorr for NlC/NlA
i_corr_nl = i0_nl

# --- Joint LinC/NlA Fit (optimize i0, alpha_a, alpha_c) ---
# Combine Tafel log data
eta_joint = np.concatenate([eta_c, eta_a])
logI_joint = np.concatenate([log_I_c, log_I_a])
# Define piecewise Tafel-domain log model for joint fit
def log_joint(eta, log_i0, alpha_a, alpha_c):
    p0 = 10**log_i0
    coeff = F/(2.303*R*T)
    # cathodic half I ≈ -i0*exp(-alpha_c*F*η/RT)
    # anodic half I ≈  i0*exp(alpha_a*F*η/RT)
    logI = np.where(
        eta<0,
        log_i0 - alpha_c*coeff*eta,
        log_i0 + alpha_a*coeff*eta
    )
    return logI
# Initial guesses: log_i0 from linear intercept, alpha_a/alpha_c from nonlinear fits
p0 = [np.log10(i_corr), alpha_a_nl, alpha_c_nl]
# Fit using curve_fit
popt_j, _ = curve_fit(lambda e, li0, aa, ac: log_joint(e, li0, aa, ac),
                      eta_joint, logI_joint, p0=p0)
log_i0_j, alpha_a_j, alpha_c_j = popt_j
# Back-calculate i0 and Tafel slopes
i0_j = 10**log_i0_j
beta_a_j = 2.303*R*T/(alpha_a_j*F)*1000
beta_c_j = 2.303*R*T/(alpha_c_j*F)*1000
# Joint model curve
logI_joint_fit = log_joint(eta_joint, log_i0_j, alpha_a_j, alpha_c_j)
# Fit metrics
tmask = np.isfinite(logI_joint_fit)&np.isfinite(logI_joint)
r2_joint = r2_score(logI_joint[tmask], logI_joint_fit[tmask])
rmse_joint = np.sqrt(mean_squared_error(logI_joint[tmask], logI_joint_fit[tmask]))

# --- Corrosion Rates ---
def calc_mmpy(icorr): return icorr * K * EW / (rho * A_cm2)
rate_lin_lin = calc_mmpy(i_corr)
rate_lin_nl = rate_lin_lin
rate_nl_nl = calc_mmpy(i_corr_nl)

# Include joint corrosion rate
rate_joint = calc_mmpy(i0_j)

# --- Plots ---
# Tafel Data as Smooth Trace and Linear Baselines
plt.figure()
# Combine cathodic and anodic Tafel data for a smooth black line
eta_comb = np.concatenate([eta_c, eta_a])
log_I_comb = np.concatenate([log_I_c, log_I_a])
# sort for smooth line
sort_idx = np.argsort(eta_comb)
plt.plot(eta_comb[sort_idx], log_I_comb[sort_idx], 'k-', label='Data')
# Cathodic baseline: extend from region start to Ecorr (η=0)
x_c_line = np.linspace(eta_c[start_c], 0, 100)
y_c_line = slope_c * x_c_line + intercept_c
plt.plot(x_c_line, y_c_line, 'b-', label=f'LinC β={beta_c:.1f} mV/dec')
# Anodic baseline: extend from Ecorr (η=0) to max anodic η
x_a_line = np.linspace(0, eta_a.max(), 100)
y_a_line = slope_a_lin * x_a_line + intercept_a_lin
plt.plot(x_a_line, y_a_line, 'r-', label=f'LinA β={beta_a_lin:.1f} mV/dec')
plt.axvline(0, linestyle='--', color='gray')
plt.axhline(np.log10(i_corr), linestyle='--', color='gray')
plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.title('Tafel Linear Fits')
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('tafel_linear_baselines.png')
# Capture y-axis limits for reuse
ymin, ymax = plt.ylim()


# Residuals Linear vs Nonlinear
plt.figure()
plt.plot(eta_c, residuals_c, 'b-', label=f'LinC Resid (RMSE={rmse_c:.2f})')
plt.plot(eta_c, y_c_nl - log_I_c, 'm--', label=f'NlC Resid (RMSE={rmse_c_nl:.2f})')
plt.plot(eta_a, residuals_a, 'r-', label=f'LinA Resid (RMSE={rmse_a:.2f})')
plt.plot(eta_a, y_a_nl - log_I_a, 'c--', label=f'NlA Resid (RMSE={rmse_a_nl:.2f})')
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('η (V)'); plt.ylabel('Residuals'); plt.title('Tafel Residuals')
plt.ylim(ymin, ymax)
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('tafel_residuals.png')

# Nonlinear Tafel Fits
plt.figure()
plt.plot(eta_c, log_I_c, 'k-', label='Data')
plt.plot(eta_c, y_c_nl, 'm--', label=f'Cath NL β={beta_c_nl:.1f} mV/dec')
plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.title('Cathodic Nonlinear Fit')
plt.ylim(ymin, ymax)
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('cathodic_nl_tafel.png')

plt.figure()
plt.plot(eta_a, log_I_a, 'k-', label='Data')
plt.plot(eta_a, y_a_nl, 'c--', label=f'Anod NL β={beta_a_nl:.1f} mV/dec')
plt.xlabel('η (V)'); plt.ylabel('log10(|I|)'); plt.title('Anodic Nonlinear Fit')
plt.ylim(ymin, ymax)
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('anodic_nl_tafel.png')

# --- Prepare Combined BV Model Curves ---
# Calculate linear-derived alphas from Tafel slopes (β in mV/dec)
alpha_c_lin = 2.303 * R * T * 1000 / (beta_c * F)
alpha_a_lin = 2.303 * R * T * 1000 / (beta_a_lin * F)

# Define full BV current function
def I_bv(eta, i0, alpha_a, alpha_c):
    return i0 * (np.exp(alpha_a * F * eta / (R * T)) - np.exp(-alpha_c * F * eta / (R * T)))
# Compute model currents and log10
eta_full = np.concatenate([eta_c, eta_a])
# LinC/LinA
I_lin_lin = I_bv(eta_full, i_corr, alpha_a_lin, alpha_c_lin)
log_lin_lin = np.log10(np.abs(I_lin_lin) + 1e-20)
# LinC/NlA (linear cathodic, nonlinear anodic)
I_lin_nl = I_bv(eta_full, i_corr, alpha_a_nl, alpha_c_lin)
log_lin_nl = np.log10(np.abs(I_lin_nl) + 1e-20)
# NlC/NlA (nonlinear cathodic, nonlinear anodic)
I_nl_nl = I_bv(eta_full, i_corr_nl, alpha_a_f, alpha_c_f)
log_nl_nl = np.log10(np.abs(I_nl_nl) + 1e-20)
# JointC/NlA (Joint linear cathodic, nonlinear anodic)
I_JC_nl = I_bv(eta_full, i0_j, alpha_a_j, alpha_c_j)
log_JC_nl = np.log10(np.abs(I_JC_nl) + 1e-20)

# Combined BV Overlay
plt.figure()
# Combined Tafel Data Trace
plt.plot(eta_comb[sort_idx], log_I_comb[sort_idx], 'k-', label='Data')
# LinC/LinA Model
plt.plot(eta_full, log_lin_lin, 'b-', label='LinC/LinA')
# LinC/NlA Model
plt.plot(eta_full, log_lin_nl, 'r--', label='LinC/NlA')
# NlC/NlA Model
plt.plot(eta_full, log_nl_nl, 'g-.', label='NlC/NlA')
# JointC/NlA overlay
plt.plot(eta_full, log_JC_nl, 'y-', label='JointC/NlA')
# Ecorr and Icorr Lines
plt.axvline(0, linestyle='--', color='gray')
plt.axhline(np.log10(i_corr), linestyle='--', color='gray')
# Labels and Limits
plt.xlabel('η (V)')
plt.ylabel('log10(|I|)')
plt.title('Combined BV Model Fits')
plt.ylim(ymin, ymax)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('tafel_combined_models.png')

# --- Compute Linear α From Tafel Slopes ---
# α derived from Tafel slope (β in mV/dec)
alpha_c_lin = 2.303 * R * T * 1000 / (beta_c * F)
alpha_a_lin = 2.303 * R * T * 1000 / (beta_a_lin * F)

# --- Summary Table ---
summary = pd.DataFrame({
    'Method': ['LinC/LinA', 'LinC/NlA', 'NlC/NlA', 'JointC/NlA'],
    'βc (mV/dec)': [beta_c, beta_c, beta_c_f, beta_c_j],
    'βa (mV/dec)': [beta_a_lin, beta_a_nl, beta_a_f, beta_a_j],
    'αc': [alpha_c_lin, alpha_c_lin, alpha_c_f, alpha_c_j],
    'αa': [alpha_a_lin, alpha_a_nl, alpha_a_f, alpha_a_j],
    'Icorr (µA)': [i_corr * 1e6, i_corr * 1e6, i_corr_nl * 1e6, i0_j*1e6],
    'Rate (mmpy)': [rate_lin_lin, rate_lin_nl, rate_nl_nl, rate_joint],
    'R²_lin': [r2_c, r2_a, np.nan, np.nan],
    'RMSE_lin': [rmse_c, rmse_a, np.nan, np.nan],
    'R²_nl': [r2_c_nl, r2_a_nl, r2_full_nl, r2_joint],
    'RMSE_nl': [rmse_c_nl, rmse_a_nl, rmse_full_nl, rmse_joint]
})

print(summary.to_string(index=False))
summary.to_csv('tafel_summary.csv', index=False)

print('✅ All analyses complete.')
