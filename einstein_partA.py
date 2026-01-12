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

    return rho_total

# --- TOV Solver ---
def tov_derivs(r, y, K, Gamma):
    """
    Solves the Tolman-Oppenheimer-Volkoff equations.
    y = [pressure, mass, nu]
    """
    p, m, nu = y
    
    if p <= 0: return [0, 0, 0] # Vacuum
    
    # Get total density
    rho = get_rho_total(p, K, Gamma)
    
    # Singularity handling at r=0
    if r == 0: return [0, 0, 0]
    
    denom = r * (r - 2*m)
    dnu_dr = 2 * (m + 4 * np.pi * r**3 * p) / denom

    # TOV Hydrostatic Equilibrium
    dp_dr = -0.5 * (rho + p) * dnu_dr
    dm_dr = 4 * np.pi * r**2 * rho
    
    return [dp_dr, dm_dr, dnu_dr]

def solve_star(rho_c_code):
    """Integrates a single star from center (rho_c) to surface (p=0)."""
    
    # Initial Conditions (Near center to avoid r=0)
    r_min = 1e-4
    
    # Central Pressure
    p_c = K_code * (rho_c_code**Gamma)
    rho_total_c = rho_c_code + p_c/(Gamma - 1)
    
    # Initial Mass (Volume * Density)
    m0 = (4/3) * np.pi * r_min**3 * rho_total_c
    
    # Initial State: [Pressure, Mass, Nu]
    # We set nu(0)=0 arbitrarily
    y0 = [p_c, m0, 0.0]
    
    # Stop condition: Pressure = 0
    def surface_event(r, y, K, G): return y[0]
    surface_event.terminal = True
    surface_event.direction = -1
    
    sol = solve_ivp(
        fun=tov_derivs,
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
        
        # Convert to Physical Units
        R_km = R_code * (L_scale / 1e5) # cm -> km
        M_sun = M_code # M_scale is M_sun
        
        return R_km, M_sun
    return None, None

# --- Main Execution  ---
if __name__ == "__main__":
    print("Generating M-R Curve for Neutron Stars...")
    
    # Sweep over central rest-mass densities
    # Typical NS range: 10^14 to 10^16 g/cm^3
    # In Code Units: 1e14/6e17 ~ 1e-4  to  1e16/6e17 ~ 0.02
    # We'll go a bit wider to see the full curve
    rho_range = np.logspace(-4, -1.2, 50) 
    
    radii = []
    masses = []
    
    for rho_c in rho_range:
        r, m = solve_star(rho_c)
        if r is not None:
            radii.append(r)
            masses.append(m)
            
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.plot(radii, masses, 'b-o', markersize=4, label='NS Solution (TOV)')
    
    plt.xlabel('Radius (km)')
    plt.ylabel(r'Mass ($M_\odot$)')
    plt.title('Neutron Star Mass-Radius Relation (TOV)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Invert X axis? Sometimes done in astro, but standard is fine.
    # We just want to see the curve.
    
    plt.savefig('ns_mass_radius.png')
    plt.show()
    
    print(f"Max Mass found: {max(masses):.4f} M_sun")