import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Geometric Units ---
# Fundamental constants (CGS)
G_cgs = 6.674e-8
c_cgs = 2.99792e10
M_sun_cgs = 1.989e33

# Geometric Scaling Factors (G = c = 1)
M_scale = M_sun_cgs                         # Mass Scale
L_scale = (G_cgs * M_scale) / (c_cgs**2)    # Length Scale
T_scale = L_scale / c_cgs                   # Time Scale
Rho_scale = M_scale / (L_scale**3)          # Density Scale
Pres_scale = Rho_scale * (c_cgs**2)         # Pressure Scale

print(f"--- Geometric Units ---")
print(f"Length Scale: {L_scale/1e5:.4f} km")
print(f"Density Scale: {Rho_scale:.4e} g/cm^3")

# --- Equation of State (EOS) Setup ---
# Parameters for Neutron Star (Read et al.)
Gamma = 1.3569
P_ref_cgs = 1.5689e31       # dyne/cm^2
rho_ref_cgs = 10.0**13      # g/cm^3

# Calculate K in CGS: P = K * rho^Gamma
K_cgs = P_ref_cgs / (rho_ref_cgs**Gamma)

# Convert K to Code Units:
K_code = K_cgs * (Rho_scale**Gamma) / Pres_scale
print(f"EOS Constant K (Code Units): {K_code:.6f}")

def get_rho_total(p, K=K_code, Gamma=Gamma):
    """
    Inverts the EOS to find Total Energy Density (rho) from Pressure (p).
    EOS: p = K * rho_r^Gamma
    rho = rho_r + p / (Gamma - 1)
    """
    if p <= 0: return 0.0
  
    rho_r = (p / K)**(1.0/Gamma)            # Rest mass density
    rho_total = rho_r + p / (Gamma - 1.0)   # Total energy density

    return rho_r, rho_total

# --- TOV Solver ---
def tov_derivs_combined(r, y, K, Gamma):
    """
    Solves the Tolman-Oppenheimer-Volkoff equations.
    y = [pressure, mass, nu, m_bary]
    """
    p, m, nu, mp = y
    
    if p <= 0: return [0, 0, 0, 0] # Vacuum
    
    # Get total density
    rho_r, rho = get_rho_total(p, K, Gamma)
    
    # Singularity handling at r=0
    if r == 0: return [0, 0, 0, 0]
    
    denom = r * (r - 2*m)
    if denom <= 0: return [0, 0, 0, 0] # Horizon safety

    # PART A
    dnu_dr = 2 * (m + 4 * np.pi * r**3 * p) / denom
    dp_dr = -0.5 * (rho + p) * dnu_dr
    dm_dr = 4 * np.pi * r**2 * rho

    # PART B
    metric_factor = (1 - 2*m/r)**(-0.5)
    dmp_dr = 4 * np.pi * r**2 * rho_r * metric_factor
    
    return [dp_dr, dm_dr, dnu_dr, dmp_dr]

def solve_star(rho_c_code):
    """Integrates a single star from center (rho_c) to surface (p=0)."""
    
    # Initial Conditions (Near center to avoid r=0)
    r_min = 1e-4
    
    # Central Pressure
    p_c = K_code * (rho_c_code**Gamma)
    rho_r_c, rho_total_c = get_rho_total(p_c)
    
    # Initial Mass (Volume * Density)
    m0 = (4/3) * np.pi * r_min**3 * rho_total_c
    mp0 = (4/3) * np.pi * r_min**3 * rho_r_c
    
    # Initial State: [Pressure, Mass, Nu]
    # We set nu(0)=0 arbitrarily
    y0 = [p_c, m0, 0.0, mp0]
    
    # Stop condition: Pressure = 0
    def surface_event(r, y, K, G): return y[0]
    surface_event.terminal = True
    surface_event.direction = -1
    
    sol = solve_ivp(
        fun=tov_derivs_combined,
        t_span=(r_min, 50.0), # Integrate far enough to find surface
        y0=y0,
        args=(K_code, Gamma),
        events=surface_event,
        rtol=1e-6, atol=1e-10
    )
    
    if sol.t_events[0].size > 0:
        # Extract Radius and Mass in Code Units
        R_code = sol.t_events[0][0]
        M_code = sol.y_events[0][0][1]
        M_bary = sol.y_events[0][0][3]
        # Convert to Physical Units
        R_km = R_code * (L_scale / 1e5) # cm -> km
        M_sun = M_code # M_scale is M_sun
        
        return R_km, M_sun, M_bary
    return None, None, None

# --- Main Execution  ---
if __name__ == "__main__":
    print("Generating M-R Curve for Neutron Stars...")
    
    # Sweep over central rest-mass densities
    # Typical NS range: 10^14 to 10^16 g/cm^3
    # In Code Units: 1e14/6e17 ~ 1e-4  to  1e16/6e17 ~ 0.02
    # We'll go a bit wider to see the full curve
    rho_range = np.logspace(-4, -0.5, 100)
    
    radii = []
    masses_grav = []
    masses_bary = []
    deltas = []
    
    for rho_c in rho_range:
        r, m, mp = solve_star(rho_c)
        if r is not None:
            radii.append(r)
            masses_grav.append(m)
            masses_bary.append(mp)
            
            # Eq 17: Fractional Binding Energy
            delta = (mp - m) / m
            deltas.append(delta)
            
    # --- Plotting ---
    # --- Plot 1: M vs R (Part A) ---
    plt.figure(figsize=(6, 4.5))
    plt.plot(radii, masses_grav, 'b-o', markersize=3)
    plt.xlabel('Radius (km)')
    plt.ylabel(r'Gravitational Mass ($M_\odot$)')
    plt.title('Neutron Star Mass-Radius Relation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/einstein_partA_MR.png')

    # Invert X axis? Sometimes done in astro, but standard is fine.
    # We just want to see the curve.
    
    # --- Plot 2: Delta vs R (Part B) ---
    plt.figure(figsize=(6, 4.5))
    plt.plot(radii, deltas, 'g-o', markersize=3)
    plt.xlabel('Radius (km)')
    plt.ylabel(r'Fractional Binding Energy $\Delta$')
    plt.title('Binding Energy vs Radius')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/einstein_partB_DeltaR.png')
    
    # --- Plot 3: M vs Mp (Part B - The Cusp) ---
    plt.figure(figsize=(6, 4.5))
    plt.plot(masses_bary, masses_grav, 'r-o', markersize=3)
    plt.xlabel(r'Baryonic Mass $M_P$ ($M_\odot$)')
    plt.ylabel(r'Gravitational Mass $M$ ($M_\odot$)')
    plt.title(r'Gravitational vs Baryonic Mass ($M$ vs $M_P$)')
    plt.grid(True, alpha=0.3)    
    plt.savefig('outputs/einstein_partB_MMp.png')

    
    # Draw a line y=x to visualize the difference
    max_val = max(max(masses_grav), max(masses_bary))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='M = Mp')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/einstein_partB_MvsMp.png')
    
    plt.show()
   
    print(f"Max Mass found: {max(masses):.4f} M_sun")
