import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Constants (CGS) ---
G = 6.674e-8
M_SUN = 1.989e33
R_EARTH = 6.371e8

# --- 2. Lane-Emden Solver (Handles singularity) ---
def lane_emden_derivs(xi, y, n):
    theta, v = y  # v = dtheta/dxi
    
    # Singularity handling at xi=0
    # At xi=0, v' = -theta^n / 3 (derived via L'Hopital limit)
    if xi == 0:
        return [0, -1.0/3.0] # theta(0)=1, so -1^n/3
    
    d_theta = v
    d_v = -2/xi * v - np.sign(theta)*(np.abs(theta)**n)
    return [d_theta, d_v]

def get_lane_emden_parameters(n):
    """
    Integrates LE equation to surface (theta=0).
    Returns: xi_surf, minus_xi2_theta_prime
    """
    # Start slightly off-center to avoid div/0 warning
    xi_eval = np.linspace(1e-5, 10, 1000) 
    
    sol = solve_ivp(
        fun=lane_emden_derivs,
        t_span=(0, 20),
        y0=[1.0, 0.0],
        args=(n,),
        # FIX IS HERE: Add 'args' or specific 'n' to the inputs
        events=lambda t, y, n: y[0], 
        rtol=1e-8, atol=1e-8
    )
    
    # The event finds the exact surface
    if not sol.t_events[0].size:
        print("Warning: Surface event not found! Integration didn't hit 0.")
        return 0, 0

    xi_surf = sol.t_events[0][0]
    theta_prime_surf = sol.y_events[0][0][1] # v at surface
    
    # Calculate the mass coefficient factor: -xi^2 * theta'
    mass_factor = - (xi_surf**2) * theta_prime_surf
    
    return xi_surf, mass_factor

# --- 3. Main Analysis ---
def analyze_low_mass_stars(csv_filename):
    # Load Data
    df = pd.read_csv(csv_filename)
    
    # Filter for low mass (e.g., M < 0.45 Solar Masses)
    low_mass = df[df['mass'] < 0.45].copy()
    
    # Convert to CGS
    M_g = low_mass['mass'] * M_SUN
    g_cgs = 10**low_mass['logg']
    R_cm = np.sqrt(G * M_g / g_cgs)
    
    low_mass['R_cm'] = R_cm
    low_mass['M_g'] = M_g

    # --- Step A: Get Lane-Emden Constants for n=1.5 (q=3) ---
    n_star = 1.5
    xi_n, omega_n = get_lane_emden_parameters(n_star)
    print(f"Lane-Emden (n={n_star}): xi_n={xi_n:.4f}, -xi^2*theta'={omega_n:.4f}")

    # --- Step B: Constrained Fit for K* ---
    # Theory: M = A * R^(-3)  (since (3-1.5)/(1-1.5) = 1.5/-0.5 = -3)
    # log10(M) = log10(A) - 3 * log10(R)
    # We fix the slope to -3 and find the best intercept A.
    
    Y = np.log10(M_g)
    X = np.log10(R_cm)
    slope_fixed = -3.0
    
    # Intercept = mean(Y - slope*X)
    intercept_log = np.mean(Y - slope_fixed * X)
    A_fit = 10**intercept_log
    
    print(f"Fit Result: M = ({A_fit:.3e}) * R^-3")
    
    # --- Step C: Solve for K* ---
    # The constant A matches the big bracket term derived from theory:
    # A = 4*pi * omega_n * [ (n+1)K / 4*pi*G ]^(n/(n-1)) * xi_n^((3-n)/(n-1))
    # Let's calculate the pieces we know.
    # Exponent for K term: n/(n-1) = 1.5/0.5 = 3
    # Exponent for xi term: (3-1.5)/(0.5) = 3
    
    # A = 4*pi * omega_n * [ (2.5 * K) / (4*pi*G) ]^3 * xi_n^3
    # Solve for K:
    # [ (2.5 * K) / (4*pi*G) ]^3 = A / (4*pi * omega_n * xi_n^3)
    # Let RHS = A / (4*pi * omega_n * xi_n^3)
    # (2.5 * K) / (4*pi*G) = RHS^(1/3)
    # K = (4*pi*G / 2.5) * RHS^(1/3)
    
    RHS = A_fit / (4 * np.pi * omega_n * xi_n**3)
    K_star = (4 * np.pi * G / (n_star + 1)) * (RHS**(1/3))
    
    print(f"Derived K* (CGS): {K_star:.4e}")

    # --- Step D: Calculate Central Density rho_c for each star ---
    # Use R = xi_n * alpha
    # R = xi_n * sqrt( (n+1)K / 4*pi*G ) * rho_c^((1-n)/2n)
    # Let Z = xi_n * sqrt( (n+1)K / 4*pi*G )
    # R = Z * rho_c^(-1/6)
    # rho_c = (R / Z)^(-6) = (Z / R)^6
    
    Z_const = xi_n * np.sqrt( (n_star+1)*K_star / (4*np.pi*G) )
    
    # Calculate rho_c for the filtered stars
    rho_c_vals = (Z_const / low_mass['R_cm'])**6
    
    # --- Step E: Plot rho_c vs Mass ---
    plt.figure(figsize=(8,6))
    plt.scatter(low_mass['mass'], rho_c_vals, color='purple', alpha=0.7)
    plt.xlabel(r'Mass ($M_\odot$)')
    plt.ylabel(r'Central Density $\rho_c$ (g/cm$^3$)')
    plt.title(r'Central Density vs Mass ($n=1.5$ fit)')
    plt.yscale('log') # Density spans orders of magnitude
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('central_density_vs_mass.png')
    plt.show()

if __name__ == "__main__":
    analyze_low_mass_stars('white_dwarf_data.csv')