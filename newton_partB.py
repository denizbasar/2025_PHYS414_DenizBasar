import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Constants (CGS System) ---
# The prompt says log(g) is in CGS, so let's stick to CGS to minimize errors.
G_CGS = 6.674e-8          # cm^3 g^-1 s^-2
M_SUN_CGS = 1.989e33      # g (Mass of Sun)
R_EARTH_CGS = 6.371e8     # cm (Radius of Earth)

def process_and_plot_data(csv_filename):
    # --- 2. Load the Data ---
    # We use pandas to read the csv. 
    # It automatically handles headers like 'wdid', 'logg', 'mass'
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_filename}'.")
        return

    print("Data loaded successfully.")
    print(df.head()) # Print first few rows to verify

    # --- 3. Physical Conversions ---
    # Data provided: 
    # 'mass' is in Solar Masses
    # 'logg' is log10(gravity) in CGS
    
    # Convert Mass to grams (CGS)
    mass_g = df['mass'] * M_SUN_CGS
    
    # Convert logg to actual gravity g (cm/s^2)
    g_cgs = 10**(df['logg'])
    
    # Calculate Radius R = sqrt(GM / g) -> Result in cm
    radius_cm = np.sqrt( (G_CGS * mass_g) / g_cgs )
    
    # Convert Radius to Earth Radii
    radius_earth = radius_cm / R_EARTH_CGS
    
    # --- 4. Plotting M vs R ---
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of the data
    # usually Mass-Radius relations have Radius on X and Mass on Y, 
    # but "M vs R" can imply M on Y. 
    plt.scatter(radius_earth, df['mass'], 
                alpha=0.5, s=10, color='black', label='Observational Data')
    
    plt.title('White Dwarf Mass-Radius Relation (Observational)')
    plt.xlabel(r'Radius ($R_\oplus$)')   # Earth Radii
    plt.ylabel(r'Mass ($M_\odot$)')      # Solar Masses
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Optional: Invert X axis? 
    # In astronomy, sometimes high gravity/small radius is on the right. 
    # But for a standard physics plot, usually we leave it normal.
    # Let's keep it standard (0 -> max) for now.
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Make sure you have the csv file in the same folder!
    process_and_plot_data('white_dwarf_data.csv')
    