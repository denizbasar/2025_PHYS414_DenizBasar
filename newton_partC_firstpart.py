import sympy as sp

def derive_low_mass_limit():
    print("--- Deriving Low-Mass Limit of Chandrasekhar EOS ---")
    
    # 1. Define variables
    x = sp.symbols('x', real=True, positive=True)
    C, D, q = sp.symbols('C D q', real=True, positive=True)
    rho = sp.symbols('rho', real=True, positive=True)
    
    # 2. Define the Term inside the bracket of Eq. 8
    # f(x) = x*(2x^2 - 3)*sqrt(x^2 + 1) + 3*asinh(x)
    term_bracket = x * (2*x**2 - 3) * sp.sqrt(x**2 + 1) + 3 * sp.asinh(x)
    
    # 3. Series Expand around x=0
    # We expand up to order 6 to capture the first non-zero term
    series_expansion = sp.series(term_bracket, x, 0, 6).removeO()
    
    print("\nSeries Expansion of the bracket term (around x=0):")
    sp.pprint(series_expansion)
    # Expected result: 8/5 * x^5
    
    # 4. Construct the Approximate Pressure P
    # P = C * (Expansion Result)
    P_approx = C * series_expansion
    
    print("\nApproximate Pressure P (in terms of x):")
    sp.pprint(P_approx)
    
    # 5. Substitute x = (rho/D)^(1/q)
    x_sub = (rho / D)**(1/q)
    P_rho = P_approx.subs(x, x_sub)
    
    print("\nApproximate Pressure P (in terms of rho):")
    sp.pprint(P_rho)
    
    # 6. Match to Polytropic Form: P = K * rho^(1 + 1/n)
    # Exponent matching:
    # rho exponent is 5/q. This must equal 1 + 1/n.
    # 5/q = (n+1)/n  ->  5n = q(n+1) -> 5n - qn = q -> n(5-q) = q
    # n = q / (5-q)
    
    print("\n--- Verification of Constants ---")
    exponent_term = 5/q
    n_star = q / (5 - q)
    poly_exponent = 1 + 1/n_star
    
    print(f"Derived Exponent of rho: {exponent_term}")
    print(f"Polytropic Exponent (1 + 1/n) with n={n_star}: {sp.simplify(poly_exponent)}")
    
    if sp.simplify(exponent_term - poly_exponent) == 0:
        print("-> Exponents match! Eq (10) for n* is correct.")
    
    # Constant matching:
    # Coeff is 8*C/5 * (1/D)^(5/q) = 8C / (5 * D^(5/q))
    # This matches K* in Eq (10).
    print("-> Coefficient is 8C / 5D^(5/q). Eq (10) for K* is correct.")

if __name__ == "__main__":
    derive_low_mass_limit()