# Stern Fit Script: Full Automated Analysis + Additional Plots
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Constants
F = 96485        # C/mol
R = 8.314        # J/mol/K
T = 298          # K
A_cm2 = 0.3167   # cm^2
EW = 7.021       # g/eq
rho = 2.33       # g/cm^3
K = 3272         # constant for mmpy conversion

# Load Data
df = pd.read_csv("LSV.csv")
E = df["Working Electrode (V)"].values
I = df["Current (mA)"].values / 1000  # A

# Apply smoothing to current data
I = savgol_filter(I, window_length=11, polyorder=3)

# Estimate Ecorr
def estimate_Ecorr(E, I):
    mask = np.abs(I) < 5e-7
    return E[mask].mean() if np.any(mask) else E[np.argmin(np.abs(I))]
Ecorr = estimate_Ecorr(E, I)

# Define Tafel region
mask_tafel = (E >= Ecorr-0.3) & (E <= Ecorr+0.3)
E_tafel = E[mask_tafel]; I_tafel = I[mask_tafel]

# Separate cathodic/anodic branches in Tafel region
mask_cath = I_tafel < 0; mask_an = I_tafel > 0
E_c = E_tafel[mask_cath]; I_c = I_tafel[mask_cath]
E_a = E_tafel[mask_an]; I_a = I_tafel[mask_an]
eta_c = E_c - Ecorr; log_I_c = np.log10(-I_c)
eta_a = E_a - Ecorr; log_I_a = np.log10(I_a)

# Automated largest linear cathodic region
best_c = {'length':0}
for i in range(len(eta_c)-5):
    for j in range(i+5, len(eta_c)):
        x, y = eta_c[i:j], log_I_c[i:j]
        slope, intercept, r, *_ = linregress(x,y)
        err = np.mean(np.abs(y - (slope*x+intercept)))
        if err<0.05 and r*r>0.995:
            if (j-i)>best_c['length']:
                best_c.update({'start':i,'end':j,'slope':slope,'intercept':intercept,'r2':r*r,'length':j-i})
start_c, end_c = best_c['start'], best_c['end']
slope_c, intercept_c = best_c['slope'], best_c['intercept']
beta_c = abs(1/slope_c)*1000  # mV/dec
i_corr = 10**intercept_c

# Automated largest linear anodic region (linear), constrained to start at Ecorr crossing
best_a = {'r2': 0, 'length': 0}
anchor_idx = np.argmin(np.abs(eta_a))  # i_corr assumed at eta = 0
window_size = 5

for end in range(anchor_idx + window_size, len(eta_a)):
    x_full = eta_a[anchor_idx:end]
    y_full = log_I_a[anchor_idx:end]

    # Try all sub-regions from full length down to minimum window
    for cutoff in range(0, len(x_full) - window_size + 1):
        x = x_full[:-cutoff] if cutoff > 0 else x_full
        y = y_full[:-cutoff] if cutoff > 0 else y_full

        slope, intercept, r, *_ = linregress(x, y)
        err = np.mean(np.abs(y - (slope * x + intercept)))
        if r**2 > best_a['r2'] and err < 0.05:
            best_a.update({
                'slope': slope,
                'intercept': intercept,
                'r2': r**2,
                'start': anchor_idx,
                'end': anchor_idx + len(x)
            })

if 'start' not in best_a or 'end' not in best_a:
    best_a['start'] = anchor_idx
    best_a['end'] = anchor_idx + window_size
    best_a['slope'] = (log_I_a[best_a['end']] - log_I_a[best_a['start']]) / (eta_a[best_a['end']] - eta_a[best_a['start']])
    best_a['intercept'] = log_I_a[best_a['start']] - best_a['slope'] * eta_a[best_a['start']]

start_a = best_a['start']
end_a = best_a['end']
slope_a_lin = best_a['slope']
intercept_a_lin = best_a['intercept']
beta_a_lin = abs(1 / slope_a_lin) * 1000  # mV/dec
start_a, end_a = best_a['start'], best_a['end']
slope_a, intercept_a = best_a['slope'], best_a['intercept']
beta_a = abs(1/slope_a) * 1000  # mV/dec
slope_a_lin, intercept_a_lin = slope_a, intercept_a  # reuse linear fit values
beta_a_lin = abs(1/slope_a_lin)*1000

# Nonlinear cathodic fit (η<0)
# Nonlinear cathodic fit on Tafel data
def bv_cath_tafel(eta, i_corr, alpha):
    return np.log10(i_corr) + (-alpha * F * eta) / (2.303 * R * T)
popt_c, _ = curve_fit(bv_cath_tafel, eta_c, log_I_c, p0=[i_corr, 0.5], maxfev=10000)
i_corr_nl_c, alpha_c_nl = popt_c
beta_c_nl = (2.303 * R * T) / (alpha_c_nl * F) * 1000
alpha_c_nl = popt_c[0]
beta_c_nl = (2.303*R*T)/(alpha_c_nl*F)*1000

# Nonlinear anodic fit constrained by i_corr (η>0)
# Nonlinear anodic fit on Tafel data (constrained by cathodic i_corr)
def bv_an_tafel(eta, alpha):
    return bv_cath_tafel(eta, i_corr_nl_c, -alpha)
popt_a, _ = curve_fit(bv_an_tafel, eta_a, log_I_a, p0=[0.5], maxfev=10000)
alpha_a_nl = popt_a[0]
beta_a_nl = (2.303 * R * T) / (alpha_a_nl * F) * 1000
alpha_a_nl = popt_a[0]
beta_a_nl = (2.303*R*T)/(alpha_a_nl*F)*1000

# Combined nonlinear BV fit (full, αa and αc)
def bv_full(eta, alpha_a, alpha_c):
    return i_corr*(np.exp((alpha_a*F*eta)/(R*T)) - np.exp((-alpha_c*F*eta)/(R*T)))
popt_f, _ = curve_fit(bv_full, E_tafel-Ecorr, I_tafel, p0=[0.3,0.3], maxfev=10000)
alpha_a_f, alpha_c_f = popt_f
beta_a_f = (2.303*R*T)/(alpha_a_f*F)*1000
beta_c_f = (2.303*R*T)/(alpha_c_f*F)*1000

# Corrosion rates (mmpy)
i_corr_density = i_corr / A_cm2
rate_lin_lin = (K * i_corr_density * EW) / rho                     # cath lin / an lin
rate_lin_nl = (K * i_corr_density * EW) / rho                     # cath lin / an nl
rate_nl_nl = (K * i_corr_density * EW) / rho                     # cath nl / an nl

# ----- Plot derivative to show transition -----
d1 = np.gradient(log_I_c, eta_c)
d2 = np.gradient(d1, eta_c)
plt.figure()
plt.plot(eta_c, d1, label='1st deriv')
plt.plot(eta_c, d2, label='2nd deriv')
# Highlight linear fit region
green_start, green_end = eta_c[start_c], eta_c[end_c]
plt.axvspan(green_start, green_end, color='green', alpha=0.3, label='Linear region')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('η (V)'); plt.ylabel('derivative'); plt.title('Cathodic Derivatives')
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('cathodic_derivatives.png')
plt.show()

# ----- Plot 1: Combined BV Model in Tafel Region -----
E_plot = E_tafel
plt.figure()
plt.plot(E_plot, np.log10(np.abs(I_tafel)), 'k.', label='Data')
E_mod = np.linspace(Ecorr-0.3,Ecorr+0.3,400)
I_mod_f = bv_full(E_mod-Ecorr, alpha_a_f, alpha_c_f)
plt.plot(E_mod, np.log10(np.abs(I_mod_f)), 'r-', label='Full BV Fit')
plt.xlim(Ecorr-0.3, Ecorr+0.3)
plt.xlabel('Potential (V)'); plt.ylabel('log10(|I|)')
plt.title('Combined BV Fit ±300 mV'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('combined_BV_fit.png')
plt.show()

# ----- Plot 2: Tafel with Baselines & Lines at Ecorr/Icorr -----
plt.figure()
plt.plot(E, np.log10(np.abs(I)), 'k.', label='Data')
# Cathodic baseline: solid from Ecorr-0.3 to Ecorr
e_c_base = np.linspace(Ecorr-0.3, Ecorr, 100)
log_c_base = slope_c*(e_c_base - Ecorr) + intercept_c
plt.plot(e_c_base, log_c_base, 'g-', linewidth=2, label='Cathodic baseline')
# Anodic baseline: solid from Ecorr to Ecorr+0.3
e_a_base = np.linspace(Ecorr, Ecorr+0.3, 100)
log_a_base = slope_a_lin*(e_a_base - Ecorr) + np.log10(i_corr)
plt.plot(e_a_base, log_a_base, 'b-', linewidth=2, label='Anodic baseline')
# Ecorr and Icorr lines
plt.axvline(Ecorr, color='black', linestyle='--', label='Ecorr')
plt.axhline(np.log10(i_corr), color='gray', linestyle='--', label='log10(i_corr)')
plt.xlim(Ecorr-0.3, Ecorr+0.3)
plt.xlabel('Potential'); plt.ylabel('log10(|I|)')
plt.title('Tafel ±300 mV with Baselines'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('tafel_baselines.png')
plt.show()

# ----- Plot 3: Cathodic nonlinear fit on Tafel plot -----
plt.figure()
plt.plot(eta_c, log_I_c, 'ko', label='Cathodic Tafel data')
log_I_c_nl = bv_cath_tafel(eta_c, i_corr_nl_c, alpha_c_nl)
plt.plot(eta_c, log_I_c_nl, 'm--', label=f'Nonlinear Fit: βc={beta_c_nl:.1f} mV/dec')
plt.xlabel('Overpotential (V)'); plt.ylabel('log10(|I|)')
plt.title('Cathodic Nonlinear BV Fit (Tafel domain)'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('cathodic_nonlinear_fit.png')
plt.show()

# ----- Plot 4: Anodic Nonlinear Fit on Tafel plot -----
plt.figure()
plt.plot(eta_a, log_I_a, 'ko', label='Anodic Tafel data')
log_I_a_nl = bv_an_tafel(eta_a, alpha_a_nl)
plt.plot(eta_a, log_I_a_nl, 'b-', label=f'Anodic Nonlinear Fit: βa={beta_a_nl:.1f} mV/dec')
plt.xlabel('Overpotential (V)'); plt.ylabel('log10(|I|)')
plt.title('Anodic Nonlinear BV Fit (i_corr constrained, Tafel domain)'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('anodic_nonlinear_fit.png')
plt.show()

# ----- Plot 5: Combined BV Fit on Tafel Plot with all models -----
plt.figure()
plt.plot(E_plot, np.log10(np.abs(I_tafel)), 'k.', label='Data')
I_lin_lin = 10**(slope_c*(E_plot[E_plot<=Ecorr]-Ecorr)+intercept_c)
I_lin_nl = 10**bv_an_tafel(E_plot[E_plot>Ecorr]-Ecorr, alpha_a_nl)
plt.plot(E_plot[E_plot<=Ecorr], np.log10(I_lin_lin), 'g--', label='LinC')
plt.plot(E_plot[E_plot>Ecorr], np.log10(I_lin_nl), 'b--', label='NlA (constrained)')
plt.plot(E_mod, np.log10(np.abs(I_mod_f)), 'r-', label='Full BV Fit')
plt.xlim(Ecorr-0.3, Ecorr+0.3)
plt.xlabel('Potential (V)'); plt.ylabel('log10(|I|)')
plt.title('All Fits: Linear & Nonlinear (Tafel domain)'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('tafel_combined_all_fits.png')
plt.show()
plt.figure()
plt.plot(E_plot, I_tafel, 'ko', label='Data')
I_nl_full = bv_full(E_plot-Ecorr, alpha_a_f, alpha_c_f)
plt.plot(E_plot, I_nl_full, 'm-', label='Full Nonlinear BV')
plt.xlabel('Potential (V)'); plt.ylabel('Current (A)')
plt.title('Nonlinear BV Fit on Tafel Region'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('full_nonlinear_BV_fit.png')
plt.show()

# ----- Plot 5: Anodic Nonlinear Fit with i_corr constraint -----
plt.figure()
plt.plot(E_a, I_a, 'ko', label='Anodic I vs E')
I_a_nl = 10**bv_an_tafel(eta_a, alpha_a_nl)
plt.plot(E_a, I_a_nl, 'b-', label=f'Anodic Nonlinear Fit: βa={beta_a_nl:.1f} mV/dec')
plt.xlabel('Potential (V)'); plt.ylabel('Current (A)')
plt.title('Anodic Nonlinear BV Fit (i_corr constrained)'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('anodic_nonlinear_fit.png')
plt.show()
plt.figure()
plt.plot(E_plot, I_tafel, 'ko', label='Data')
I_nl_full = bv_full(E_plot-Ecorr, alpha_a_f, alpha_c_f)
plt.plot(E_plot, I_nl_full, 'm-', label='Full Nonlinear BV')
plt.xlabel('Potential (V)'); plt.ylabel('Current (A)')
plt.title('Nonlinear BV Fit on Tafel Region'); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('full_nonlinear_BV_fit.png')
plt.show()

# ----- Table of Results -----
results = pd.DataFrame({
    'Method': ['LinC/LinA','LinC/NlA','NlC/NlA'],
    'βc (mV/dec)': [beta_c, beta_c, beta_c_f],
    'βa (mV/dec)': [beta_a_lin, beta_a_nl, beta_a_f],
    'αa': [np.nan, alpha_a_nl, alpha_a_f],
    'αc': [np.nan, alpha_c_nl, alpha_c_f],
    'Icorr (µA)': [i_corr*1e6, i_corr*1e6, i_corr*1e6],
    'Corrosion Rate (mmpy)': [rate_lin_lin, rate_lin_nl, rate_nl_nl]
})
# Save results table as CSV and display
results.to_csv('tafel_analysis_results.csv', index=False)
print(results.to_string(index=False))
