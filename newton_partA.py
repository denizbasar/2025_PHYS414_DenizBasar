import sympy as sp

def verify_coefficients():
    """
    Symbolically verifies the Taylor series expansion of the Lane-Emden equation.
    """
    # 1. Setup
    xi = sp.symbols('xi')
    n = sp.symbols('n', real=True)
    
    # 2. Define the series ansatz (up to order 4)
    # We want to FIND a2 and a4 to check if they match the textbook.
    a2, a4 = sp.symbols('a2 a4')
    theta_approx = 1 + a2*xi**2 + a4*xi**4
    
    # 3. Define the ODE LHS: (1/xi^2)*(xi^2*theta')' + theta^n
    # We substitute our approximation into this ODE.
    term1 = sp.diff(xi**2 * sp.diff(theta_approx, xi), xi)
    ODE_lhs = (1/xi**2) * term1 + theta_approx**n
    
    # 4. Expand in series to isolate powers of xi
    # .series(x, 0, n) expands around x=0 up to order n
    series_res = sp.series(ODE_lhs, xi, 0, 5)
    
    # 5. Extract coefficients
    # The 'removeO()' removes the big-O notation so we can do math
    clean_series = series_res.removeO()
    
    const_term = clean_series.subs(xi, 0)
    xi2_term   = clean_series.coeff(xi, 2)
    
    print("-" * 40)
    print("VERIFICATION OF SERIES COEFFICIENTS")
    print("-" * 40)
    
    # Solve for a2 (Constant term must be 0)
    # Eq: 6*a2 + 1 = 0
    calculated_a2 = sp.solve(const_term, a2)[0]
    print(f"Coeff of xi^0 is: {const_term}")
    print(f"Solution for a2:  {calculated_a2}  (Textbook says -1/6)")
    
    # Solve for a4 (xi^2 term must be 0)
    # Eq involves a2 and a4. Plug in our calculated a2.
    eq_a4 = xi2_term.subs(a2, calculated_a2)
    calculated_a4 = sp.solve(eq_a4, a4)[0]
    print(f"Coeff of xi^2 is: {xi2_term}")
    print(f"Solution for a4:  {calculated_a4}   (Textbook says n/120)")
    print("-" * 40)

if __name__ == "__main__":
    verify_coefficients()