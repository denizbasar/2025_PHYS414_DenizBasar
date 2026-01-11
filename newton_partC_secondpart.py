import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
G_CGS = 6.674e-8
M_SUN_CGS = 1.989e33
R_EARTH_CGS = 6.371e8

def fit_white_dwarf_parameters(csv_filename):
    # 1. Load and Process Data (Same as Part b)
    df = pd.read_csv(csv_filename)
    mass_g = df['mass'] * M_SUN_CGS
    g_cgs = 10**(df['logg'])
    radius_cm = np.sqrt((G_CGS * mass_g) / g_cgs)
    
    # Store in DataFrame for easier filtering
    df['R_cm'] = radius_cm
    df['M_g'] = mass_g
    df['R_earth'] = radius_cm / R_EARTH_CGS
    
    # 2. Filter for Low Mass Stars
    # The prompt says Eq 9 only holds for low mass.
    # Let's try a cutoff, e.g., Mass < 0.4 Solar Masses
    low_mass_df = df[df['mass'] < 0.4]
    
    print(f"Using {len(low_mass_df)} stars for the low-mass fit.")
    
    # 3. Log-Log Fit
    # We want slope of log(M) vs log(R)
    log_m = np.log10(low_mass_df['M_g'])
    log_r = np.log10(low_mass_df['R_cm'])
    
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_m)
    
    print(f"\n--- Fit Results ---")
    print(f"Slope (log M vs log R): {slope:.4f}")
    
    # 4. Calculate n and q
    # Slope = (3 - n) / (1 - n)
    # slope * (1 - n) = 3 - n
    # slope - slope*n = 3 - n
    # slope - 3 = slope*n - n = n(slope - 1)
    # n = (slope - 3) / (slope - 1)
    
    n_fit = (slope - 3) / (slope - 1)
    print(f"Derived Polytropic Index n: {n_fit:.4f}")
    
    # Calculate q using n = q / (5 - q)
    # n(5 - q) = q
    # 5n - nq = q
    # 5n = q(1 + n)
    # q = 5n / (1 + n)
    
    q_fit = 5 * n_fit / (1 + n_fit)
    print(f"Derived q value: {q_fit:.4f}")
    
    # 5. Snap to Integer (Hint says assume integer)
    q_integer = round(q_fit)
    print(f"Integer q: {q_integer}")
    
    # Recalculate n for the integer q (likely q=3 -> n=1.5)
    n_final = q_integer / (5 - q_integer)
    print(f"Final n (using integer q): {n_final}")

    # 6. Plot to confirm linearity (The "Good Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(log_r, log_m, label='Low Mass Data', alpha=0.6)
    
    # Plot the fit line
    fit_line = slope * log_r + intercept
    plt.plot(log_r, fit_line, color='red', label=f'Fit: slope={slope:.2f}')
    
    plt.xlabel(r'$\log_{10}(R)$ [cm]')
    plt.ylabel(r'$\log_{10}(M)$ [g]')
    plt.title('Log-Log Plot: Testing Power Law Relation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('log_log_fit.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    fit_white_dwarf_parameters('white_dwarf_data.csv')