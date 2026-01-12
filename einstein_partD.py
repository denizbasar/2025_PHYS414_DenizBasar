import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# --- Constants & Units ---
G_cgs = 6.674e-8
c_cgs = 2.99792e10
M_sun_cgs = 1.989e33

# Geometric Scaling Factors
M_scale = M_sun_cgs
L_scale = (G_cgs * M_scale) / (c_cgs**2)
Rho_scale = M_scale / (L_scale**3)
Pres_scale = Rho_scale * (c_cgs**2)

# Base EOS Parameters
Gamma = 1.3569
P_ref_cgs = 1.5689e31
rho_ref_cgs = 10.0**13
K_cgs_base = P_ref_cgs / (rho_ref_cgs**Gamma)
K_code_base = K_cgs_base * (Rho_scale**Gamma) / Pres_scale

# --- Physics Engine ---
def get_rho_data(p, K, Gamma=Gamma):
    if p <= 0: return 0.0, 0.0
    rho_r = (p / K)**(1.0/Gamma)
    rho_total = rho_r + p / (Gamma - 1.0)
    return rho_r, rho_total

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

def find_max_mass_for_K(K_val):
    # Sweep density to find max mass
    rho_range = np.logspace(-5, -0.5, 30) 
    local_masses = []
    
    for rho_c in rho_range:
        r_min = 1e-4
        p_c = K_val * (rho_c**Gamma)
        rho_r_c, rho_total_c = get_rho_data(p_c, K_val)
        
        m0 = (4/3) * np.pi * r_min**3 * rho_total_c
        y0 = [p_c, m0, 0.0]
        
        def surface_event(r, y, K, G): return y[0]
        surface_event.terminal = True
        surface_event.direction = -1
        
        # --- FIX: INCREASE INTEGRATION LIMIT ---
        # Increased from 100.0 to 1000.0 to handle "puffy" high-K stars
        sol = solve_ivp(
            tov_derivs, (r_min, 1000.0), y0, 
            args=(K_val, Gamma), events=surface_event,
            rtol=1e-5, atol=1e-8
        )
        
        if sol.t_events[0].size > 0:
            local_masses.append(sol.y_events[0][0][1])
            
    if len(local_masses) == 0: return 0.0
    return max(local_masses)

# --- Main Sweep ---
if __name__ == "__main__":
    if not os.path.exists('outputs'): os.makedirs('outputs')
    print("Running Parameter Sweep on K...")
    
    # Sweep K from 0.1 to 3.0 (Code Units)
    # The plot showed the limit is likely around K=1.2 or 1.5, so 3.0 is enough.
    k_values = np.linspace(0.1, 3.0, 20)
    
    max_masses = []
    for k_val in k_values:
        m_max = find_max_mass_for_K(k_val)
        max_masses.append(m_max)
        print(f"K={k_val:.3f} -> Max Mass: {m_max:.3f}")

    # Interpolate
    target_mass = 2.5
    k_values = np.array(k_values)
    max_masses = np.array(max_masses)
    
    required_K = np.interp(target_mass, max_masses, k_values)
    print(f"\nLimit K for {target_mass} M_sun: {required_K:.6f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, max_masses, 'b-o', linewidth=2)
    plt.axhline(y=2.5, color='r', linestyle='--', label='Observed Max ($2.5 M_\odot$)')
    plt.axvline(x=required_K, color='k', linestyle=':', label=f'Limit: $K \\approx {required_K:.3f}$')
    plt.scatter([required_K], [2.5], color='red', zorder=5)
    
    plt.xlabel('Polytropic Constant $K$ (Code Units)')
    plt.ylabel(r'Maximum Stable Mass ($M_\odot$)')
    plt.title('Effect of EOS Stiffness ($K$) on Maximum Neutron Star Mass')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/einstein_partD_K_Sweep.png')
    plt.show()