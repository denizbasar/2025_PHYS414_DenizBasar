import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# --- 1. Constants (CGS) ---
G = 6.674e-8
M_SUN = 1.989e33
R_EARTH = 6.371e8

# Fundamental Constants for Theoretical Check (Eq 11)
h_bar = 1.05457e-27  # erg s
m_e = 9.10938e-28    # g
c = 2.99792e10       # cm/s
m_u = 1.66054e-24    # g (atomic mass unit)
mu_e = 2.0           # Nucleons per electron

# Value of K* derived in Part C (Update this if your previous run gave a slightly different number)
K_star = 2.7926e12 
q = 3.0

# --- 2. Physics Engines ---

def get_C_from_D(D):
    """Calculates C using the relation derived in Part C."""
    # C = (5/8) * K* * D^(5/3)
    return (5.0 * K_star * (D**(5.0/3.0))) / 8.0

def chandrasekhar_derivs(r, y, C, D):
    """The Full Chandrasekhar ODEs (solving for density rho directly)."""
    m, rho = y
    
    if rho <= 0: return [0, 0] # Safety
    
    # x = (rho/D)^(1/3)
    x = (rho / D)**(1.0/3.0)
    
    # dP/drho derived analytically
    dP_drho = (8.0 * C / (3.0 * D)) * (x**2) / np.sqrt(x**2 + 1)
    
    # Mass conservation
    dmdr = 4 * np.pi * r**2 * rho
    
    # Hydrostatic Equilibrium with Chain Rule
    if r == 0:
        drhodr = 0
    else:
        dP_dr = - (G * m * rho) / r**2
        drhodr = dP_dr / dP_drho
        
    return [dmdr, drhodr]

def solve_one_star(rho_c, C, D):
    """Integrates a single star from center (rho_c) to surface (rho=0)."""
    r_min = 100.0 # Start small
    m0 = (4/3) * np.pi * r_min**3 * rho_c
    y0 = [m0, rho_c]
    
    # Stop when density hits 0
    surface_event = lambda t, y, C, D: y[1]
    surface_event.terminal = True
    surface_event.direction = -1
    
    sol = solve_ivp(
        fun=chandrasekhar_derivs,
        t_span=(r_min, 2e9), # Max radius 20,000 km
        y0=y0,
        args=(C, D),
        events=surface_event,
        rtol=1e-5, atol=1e-8
    )
    
    if sol.t_events[0].size > 0:
        R = sol.t_events[0][0]
        M = sol.y_events[0][0][0]
        return R, M
    return np.nan, np.nan

def generate_model_curve(D, num_points=25):
    """Generates the master curve for a given D."""
    C = get_C_from_D(D)
    # Sweep densities from low (10^5) to very high (10^10)
    rho_vals = np.logspace(5, 9.5, num_points)
    
    R_list, M_list = [], []

    for rho in rho_vals:
        r, m = solve_one_star(rho, C, D)
        if not np.isnan(r):
            R_list.append(r)
            M_list.append(m)

            
    # Sort by Radius for Interpolation (Radius decreases as mass increases)
    R_arr = np.array(R_list)
    M_arr = np.array(M_list)
    idx = np.argsort(R_arr) 
    return R_arr[idx], M_arr[idx]

# --- 3. The Optimizer ---

def objective_function(D, df_obs):
    """Calculates Total Squared Error for a trial D."""
    # 1. Generate Theory Curve
    R_model, M_model = generate_model_curve(D)

    # Safety: Need enough points
    if len(R_model) < 5: return 1e99
    
    # 2. Cubic Spline Interpolation (The 's' word!)
    # We predict Mass given Radius
    spline = CubicSpline(R_model, M_model, extrapolate=False)
    
    # 3. Calculate Residuals
    # We only fit stars that fall within our model's radius range
    valid_mask = (df_obs['R_cm'] >= R_model.min()) & (df_obs['R_cm'] <= R_model.max())
    valid_data = df_obs[valid_mask]
    
    M_pred = spline(valid_data['R_cm'])
    residuals = valid_data['M_g'] - M_pred
    
    error = np.sum(residuals**2)
    print(f"Trial D={D:.2e} -> Error={error:.2e}")
    return error

# --- 4. Main Execution ---

def run_analysis(csv_file):
    # Load Data
    df = pd.read_csv(csv_file)
    df['M_g'] = df['mass'] * M_SUN
    df['g_cgs'] = 10**df['logg']
    df['R_cm'] = np.sqrt(G * df['M_g'] / df['g_cgs'])
    
    print("--- Starting Optimization for Parameter D ---")
    
    # We search for D in the range suggested by theory (approx 10^6 - 10^7)
    # Using bounded minimization
    res = minimize_scalar(
        objective_function, 
        bounds=(5e5, 5e6), 
        args=(df,),
        method='bounded'
    )
    
    best_D = res.x
    best_C = get_C_from_D(best_D)
    
    print("\n" + "="*40)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best Fit D: {best_D:.4e} g/cm^3")
    print(f"Corresp. C: {best_C:.4e}")
    print("="*40)
    
    # --- Compare with Theory (Eq 11) ---
    D_theory = (m_u * m_e**3 * c**3 * mu_e) / (3 * np.pi**2 * h_bar**3)
    C_theory = (m_e**4 * c**5) / (24 * np.pi**2 * h_bar**3)
    
    print(f"\nTheoretical D: {D_theory:.4e}")
    print(f"Theoretical C: {C_theory:.4e}")
    print(f"Ratio D_fit/D_theory: {best_D/D_theory:.3f}")
    
    # --- Final Plotting ---
    R_fit, M_fit = generate_model_curve(best_D, num_points=50)
    
    plt.figure(figsize=(10, 7))
    
    # Observational Data
    plt.scatter(df['R_cm']/R_EARTH, df['mass'], 
                color='black', alpha=0.4, s=15, label='Observations')
    
    # Best Fit Curve
    plt.plot(R_fit/R_EARTH, M_fit/M_SUN, 
             color='red', linewidth=2, label=f'Best Fit (D={best_D:.2e})')
    
    plt.xlabel(r'Radius ($R_\oplus$)')
    plt.ylabel(r'Mass ($M_\odot$)')
    plt.title('Final Model: Full Chandrasekhar EOS Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Invert X axis for standard Astro style (optional, but standard)
    # plt.gca().invert_xaxis() 
    
    plt.savefig('final_fit_plot.png')
    plt.show()

if __name__ == "__main__":
    run_analysis('white_dwarf_data.csv')