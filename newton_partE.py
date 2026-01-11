import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp

# --- 1. Constants (CGS) ---
G = 6.674e-8
M_SUN = 1.989e33
h_bar = 1.05457e-27
m_e = 9.10938e-28
c = 2.99792e10
m_u = 1.66054e-24
mu_e = 2.0
pi = np.pi

# Best fit D from Part D
D_fit = 4.77e6 
# Calculate corresponding C
K_star_partC = 2.7926e12
C_fit = (5.0 * K_star_partC * (D_fit**(5.0/3.0))) / 8.0

# --- 2. Symbolic Proof for n=3 ---
def prove_relativistic_limit():
    print("--- 1. Symbolic Proof: High-Density Limit ---")
    x = sp.symbols('x', real=True, positive=True)
    C = sp.symbols('C', real=True, positive=True)
    
    # The term in brackets from Eq. 8
    bracket = x * (2*x**2 - 3) * sp.sqrt(x**2 + 1) + 3 * sp.asinh(x)
    
    # We want the limit as x -> infinity
    # Dominant term behavior:
    # sqrt(x^2+1) -> x
    # 2x^2 * x * x = 2x^4
    # Let's ask SymPy for the leading term at infinity
    limit_expr = sp.limit(bracket / x**4, x, sp.oo)
    
    print(f"Limit of Bracket / x^4 as x->oo: {limit_expr}")
    print(f"Therefore, P approx {limit_expr} * C * x^4")
    
    # P ~ x^4. Since x ~ rho^(1/3), P ~ rho^(4/3)
    # Polytrope P ~ rho^(1 + 1/n)
    # 4/3 = 1 + 1/n  -> 1/3 = 1/n -> n = 3
    print("Since P ~ rho^(4/3), this implies n = 3 (Relativistic Polytrope).")

# --- 3. Numerical Calculation of M_Ch ---
def chandrasekhar_derivs(r, y, C, D):
    m, rho = y
    if rho <= 0: return [0, 0]
    x = (rho / D)**(1.0/3.0)
    dP_drho = (8.0 * C / (3.0 * D)) * (x**2) / np.sqrt(x**2 + 1)
    dmdr = 4 * np.pi * r**2 * rho
    if r == 0: drhodr = 0
    else: drhodr = - (G * m * rho) / (r**2 * dP_drho)
    return [dmdr, drhodr]

def get_max_mass_numerical():
    print("\n--- 2. Numerical Maximum Mass ---")
    # We integrate a star with INSANE central density to approximate infinity
    rho_c_extreme = 1e14 # Extremely high density
    
    y0 = [0, rho_c_extreme] # Mass 0, Rho_c
    
    # Stop when rho=0
    event = lambda t, y: y[1]
    event.terminal = True
    event.direction = -1
    
    # Need to start slightly off-zero r
    r_min = 1.0
    m_min = (4/3) * pi * r_min**3 * rho_c_extreme
    y0 = [m_min, rho_c_extreme]
    
    sol = solve_ivp(
        fun=lambda r, y: chandrasekhar_derivs(r, y, C_fit, D_fit),
        t_span=(r_min, 1e9),
        y0=y0,
        events=event,
        rtol=1e-6, atol=1e-8
    )
    
    M_max_numerical = sol.y_events[0][0][0]
    print(f"Numerical M_Ch (using D={D_fit:.2e}): {M_max_numerical/M_SUN:.4f} Solar Masses")
    return M_max_numerical

# --- 4. Theoretical Calculation ---
def calculate_theoretical_MCh():
    print("\n--- 3. Theoretical Calculation ---")
    # For n=3 polytrope, M is independent of rho_c!
    # M = 4*pi * ( (n+1)K / 4*pi*G )^1.5 * xi_3^2 * |theta'_3|
    # For n=3, Lane-Emden constants are roughly:
    # xi_3 = 6.89685
    # xi_3^2 * |theta'_3| = 2.01824
    
    mass_factor_n3 = 2.01824
    
    # We need K for the relativistic limit (n=3)
    # From proof above: P = 2C * x^4 = 2C * (rho/D)^(4/3)
    # So K_rel = 2C / D^(4/3)
    K_rel = 2 * C_fit / (D_fit**(4.0/3.0))
    
    term = ((3 + 1) * K_rel) / (4 * pi * G)
    M_theory = 4 * pi * (term**1.5) * mass_factor_n3
    
    print(f"Theoretical M_Ch (derived from D): {M_theory/M_SUN:.4f} Solar Masses")
    
    # Textbook Formula (Ch. 19 KWW)
    # M_Ch = 5.83 / mu_e^2
    M_textbook = (5.83 / mu_e**2) * M_SUN
    print(f"Textbook M_Ch (5.83/mu_e^2): {M_textbook/M_SUN:.4f} Solar Masses")

if __name__ == "__main__":
    prove_relativistic_limit()
    get_max_mass_numerical()
    calculate_theoretical_MCh()