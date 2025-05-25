# Exhaustive anodic model comparison with voltage window around Ecorr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
F = 96485
R = 8.314
T = 298
A_cm2 = 0.3167
EW = 7.021
rho = 2.33
K = 3272

# Voltage window relative to Ecorr (V)
V_window = 0.100  # +/- 100 mV

def corrosion_rate(i_corr):
    return (K * i_corr * EW) / rho

# Models
def M1(E, i0, alpha, Ecorr):
    eta = E - Ecorr
    return i0 * np.exp((alpha * F * eta) / (R * T))

def M2(E, i0, alpha, Ecorr, ilim):
    eta = E - Ecorr
    i_bv = i0 * np.exp((alpha * F * eta) / (R * T))
    return (i_bv * ilim) / (i_bv + ilim)

def M3(E, i0, alpha, Ecorr, R_ohm):
    i_bv = i0 * np.exp((alpha * F * (E - Ecorr)) / (R * T))
    eta = E - Ecorr - R_ohm * i_bv
    return i0 * np.exp((alpha * F * eta) / (R * T))

def M4(E, i0, alpha, Ecorr, ilim, R_ohm):
    i_bv = i0 * np.exp((alpha * F * (E - Ecorr)) / (R * T))
    i_lim_term = (i_bv * ilim) / (i_bv + ilim)
    eta = E - Ecorr - R_ohm * i_lim_term
    return (i0 * np.exp((alpha * F * eta) / (R * T)) * ilim) / (i0 * np.exp((alpha * F * eta) / (R * T)) + ilim)

# Load data
file = "LSV.csv"
df = pd.read_csv(file)
E = df["Working Electrode (V)"].values
I_raw = df["Current (mA)"].values

if np.max(np.abs(I_raw)) > 0.01:
    print("Scaling current from mA to A")
    I = I_raw / 1000
else:
    I = I_raw

anodic_mask = I > 0
E_an_full = E[anodic_mask]
I_an_full = I[anodic_mask]

# Precompute Ecorr using midpoint or where current starts rising
Ecorr_guess = E_an_full[np.argmax(np.gradient(I_an_full))]

# Apply voltage window around Ecorr_guess
window_mask = (E_an_full >= Ecorr_guess - V_window) & (E_an_full <= Ecorr_guess + V_window)
E_an = E_an_full[window_mask]
I_an = I_an_full[window_mask]
E_plot = np.linspace(min(E_an), max(E_an), 400)

results = []

# Fit each model
models = {
    "M1 (Simple BV)": (M1, [1e-6, 0.5, Ecorr_guess], ([1e-12, 0.01, min(E_an)], [1e-2, 1.0, max(E_an)])),
    "M2 (Mixed Control)": (M2, [1e-6, 0.5, Ecorr_guess, 1e-3], ([1e-12, 0.01, min(E_an), 1e-6], [1e-2, 1.0, max(E_an), 1e2])),
    "M3 (Ohmic Drop)": (M3, [1e-6, 0.5, Ecorr_guess, 1.0], ([1e-12, 0.01, min(E_an), 0], [1e-2, 1.0, max(E_an), 100])),
    "M4 (Full Model)": (M4, [1e-6, 0.5, Ecorr_guess, 1e-3, 1.0], ([1e-12, 0.01, min(E_an), 1e-6, 0], [1e-2, 1.0, max(E_an), 1e2, 100]))
}

fits = {}

for label, (model_func, p0, bounds) in models.items():
    try:
        popt, _ = curve_fit(model_func, E_an, I_an, p0=p0, bounds=bounds, maxfev=20000)
        I_fit = model_func(E_plot, *popt)
        residuals = I_an - model_func(E_an, *popt)
        chi2 = np.sum((residuals) ** 2)
        i0 = popt[0]
        i_corr = i0 / A_cm2
        mmpy = corrosion_rate(i_corr)
        results.append((label, popt, chi2, i_corr, mmpy))
        fits[label] = (E_plot, I_fit)
    except Exception as e:
        print(f"Fit failed for {label}: {e}")

# Print results
print("\n--- Anodic Model Comparison (Windowed Around Ecorr) ---")
for label, popt, chi2, i_corr, mmpy in results:
    print(f"{label}: chi2 = {chi2:.3e}, i_corr = {i_corr:.2e} A/cm², mmpy = {mmpy:.3f}")
    print(f"  Params: {['%.3e' % p for p in popt]}\n")

# Plot
plt.figure()
plt.plot(E_an, I_an, 'ko', label='Data')
for label, (E_fit, I_fit) in fits.items():
    if np.all(np.isfinite(I_fit)):
        plt.plot(E_fit, I_fit, label=label)
plt.xlabel("Potential (V)")
plt.ylabel("Current (A)")
plt.title(f"Anodic Model Fit ±{int(V_window*1000)} mV Around Ecorr")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("anodic_model_windowed_fit.png")
print("Saved: anodic_model_windowed_fit.png")
