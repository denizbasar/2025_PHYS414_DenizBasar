import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# --- 1. Constants & EOS Setup ---
G_cgs = 6.674e-8
c_cgs = 2.99792e10
M_sun_cgs = 1.989e33

# Geometric Scaling Factors
M_scale = M_sun_cgs
L_scale = (G_cgs * M_scale) / (c_cgs**2)
Rho_scale = M_scale / (L_scale**3)
Pres_scale = Rho_scale * (c_cgs**2)

# EOS Parameters
Gamma = 1.3569
P_ref_cgs = 1.5689e31
rho_ref_cgs = 10.0**13
K_cgs = P_ref_cgs / (rho_ref_cgs**Gamma)
K_code = K_cgs * (Rho_scale**Gamma) / Pres_scale

def get_rho_data(p, K=K_code, Gamma=Gamma):
    if p <= 0: return 0.0, 0.0
    rho_r = (p / K)**(1.0/Gamma)
    rho_total = rho_r + p / (Gamma - 1.0)
    return rho_r, rho_total

# --- 2. TOV Solver ---
def tov_derivs(r, y, K, Gamma):
    p, m, nu = y
    if p <= 0: return [0, 0, 0]
    
    rho_r, rho = get_rho_data(p, K, Gamma)
    if r == 0: return [0, 0, 0]
    
    denom = r * (r - 2*m)
    if denom <= 0: return [0, 0, 0]
    
    dnu_dr = 2 * (m + 4 * np.pi * r**3 * p) / denom
    dp_dr = -0.5 * (rho + p) * dnu_dr
    dm_dr = 4 * np.pi * r**2 * rho
    
    return [dp_dr, dm_dr, dnu_dr]

def solve_star_mass_only(rho_c_code):
    """Simplified solver that just returns Mass."""
    r_min = 1e-4
    p_c = K_code * (rho_c_code**Gamma)
    rho_r_c, rho_total_c = get_rho_data(p_c)
    
    m0 = (4/3) * np.pi * r_min**3 * rho_total_c
    y0 = [p_c, m0, 0.0]
    
    def surface_event(r, y, K, G): return y[0]
    surface_event.terminal = True
    surface_event.direction = -1
    
    sol = solve_ivp(
        fun=tov_derivs,
        t_span=(r_min, 50.0),
        y0=y0,
        args=(K_code, Gamma),
        events=surface_event,
        rtol=1e-6, atol=1e-10
    )
    
    if sol.t_events[0].size > 0:
        return sol.y_events[0][0][1] # Gravitational Mass
    return None

# --- 3. Stability Analysis ---
if __name__ == "__main__":
    if not os.path.exists('outputs'): os.makedirs('outputs')
    
    print("Running Stability Analysis...")
    
    # 1. Generate Data
    # We sweep a slightly wider range to see the "fall off" clearly
    rho_range = np.logspace(-4, -0.5, 150)
    masses = []
    valid_rhos = []
    
    for rho in rho_range:
        m = solve_star_mass_only(rho)
        if m is not None:
            masses.append(m)
            valid_rhos.append(rho)
            
    masses = np.array(masses)
    valid_rhos = np.array(valid_rhos)
    
    # 2. Find the Maximum Mass (The Tipping Point)
    idx_max = np.argmax(masses)
    max_mass = masses[idx_max]
    rho_crit = valid_rhos[idx_max]
    
    print(f"Maximum Stable Mass: {max_mass:.5f} M_sun")
    print(f"Critical Central Density: {rho_crit:.5f} (Code Units)")
    
    # 3. Split into Stable and Unstable Branches
    # Stable: Low density up to peak
    rho_stable = valid_rhos[:idx_max+1]
    M_stable   = masses[:idx_max+1]
    
    # Unstable: Peak to High density
    rho_unstable = valid_rhos[idx_max:]
    M_unstable   = masses[idx_max:]
    
    # 4. Convert Density to Physical Units for Plotting
    # Code Density -> g/cm^3
    rho_phys_stable = rho_stable * Rho_scale
    rho_phys_unstable = rho_unstable * Rho_scale
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    
    # Plot Stable Branch (Solid Blue)
    plt.plot(rho_phys_stable, M_stable, 'b-', linewidth=2, label='Stable Branch')
    
    # Plot Unstable Branch (Dashed Red)
    plt.plot(rho_phys_unstable, M_unstable, 'r--', linewidth=2, label='Unstable Branch')
    
    # Mark the Max Mass
    plt.scatter([rho_crit * Rho_scale], [max_mass], color='k', zorder=5)
    plt.text(rho_crit * Rho_scale, max_mass + 0.001, f' Max Mass: {max_mass:.3f} $M_\odot$', 
             ha='center', fontsize=10)
    
    plt.xscale('log')
    plt.xlabel(r'Central Density $\rho_c$ (g/cm$^3$)')
    plt.ylabel(r'Gravitational Mass ($M_\odot$)')
    plt.title('Neutron Star Stability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    plt.savefig('outputs/einstein_partC_Stability.png')
    plt.show()