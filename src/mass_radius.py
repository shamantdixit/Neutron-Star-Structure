"""
Calculates the neutron star mass-radius relationship using a simplified
hybrid Fermi Gas Equation of State.

This script solves the TOV equations over a wide range of central pressures
to generate a full M-R curve and identify the maximum possible stellar mass.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy import constants as const

# --- Physical Constants (CGS units) ---
G = const.G.cgs.value       # Gravitational constant (cm^3 g^-1 s^-2)
c = const.c.cgs.value       # Speed of light (cm s^-1)
M_sun = const.M_sun.cgs.value  # Solar mass (g)
CM_PER_KM = 1e-5            # Conversion factor from cm to km

# --- Equation of State (EoS) ---
# A simplified Fermi Gas model for degenerate neutron matter.
# The characteristic energy density scale is derived from fundamental constants.
eps0 = 5.346e36  # erg cm^-3 (nuclear saturation energy density)

# EoS is modeled as a sum of non-relativistic and relativistic components.
A_NR = 2.4216  # Coefficient for the non-relativistic term
A_R = 2.8663   # Coefficient for the ultra-relativistic term


def eps_bar(p_bar):
    """Calculates dimensionless energy density from dimensionless pressure."""
    return A_NR * p_bar**(3/5) + A_R * p_bar


def eps_phys(p_phys):
    """Calculates physical energy density (erg/cm^3) from physical pressure."""
    p_bar = p_phys / eps0  # Normalize pressure
    return eps0 * eps_bar(p_bar)


def tov_equations(r, y):
    """
    Defines the Tolman-Oppenheimer-Volkoff (TOV) equations for GR.

    Args:
        r (float): Radius in cm.
        y (list): State vector [pressure, enclosed_mass].

    Returns:
        list: Derivatives [dP/dr, dM/dr].
    """
    p, M = y
    if p <= 0.0:
        return [0.0, 0.0]

    eps = eps_phys(p)
    rho = eps / c**2

    # These factors represent the corrections from General Relativity.
    num = G * rho * M / r**2
    fac1 = 1.0 + p / eps                      # Special relativity correction
    fac2 = 1.0 + 4.0 * np.pi * r**3 * p / (M * c**2)  # Curvature correction
    fac3 = 1.0 - 2.0 * G * M / (c**2 * r)     # Redshift correction

    # The full GR pressure gradient.
    dpdr = -num * fac1 * fac2 / fac3
    # Mass continuity is the same, but rho is mass-energy density.
    dMdr = 4.0 * np.pi * r**2 * rho

    return [dpdr, dMdr]


# --- Surface Detection Event ---
def surface(r, y):
    """Event function to stop integration when the surface is reached."""
    # The surface is where pressure drops to near zero.
    return y[0] - 1.0e-8 * eps0
surface.terminal, surface.direction = True, -1  # Stop when event is found


def integrate_tov(p_c_bar, r_start=1e-2, r_max=5e6):
    """
    Integrates the TOV equations for a single star with a given central pressure.

    Args:
        p_c_bar (float): Dimensionless central pressure (in units of eps0).
        r_start (float): Small initial radius to avoid r=0 singularity.
        r_max (float): Maximum integration radius (cm).

    Returns:
        tuple: (Radius in km, Mass in solar masses).
    """
    # Initialize with central conditions.
    p_c = p_c_bar * eps0
    eps_c = eps_phys(p_c)
    rho_c = eps_c / c**2

    # Initial mass inside the small core.
    M_start = (4.0/3.0) * np.pi * r_start**3 * rho_c
    y0 = [p_c, M_start]

    # Solve the ODE system.
    sol = solve_ivp(
        tov_equations, (r_start, r_max), y0,
        events=surface, rtol=1e-7, atol=1e-9, max_step=1e5
    )

    # Extract surface values from the event data.
    if sol.t_events[0].size > 0:
        R_cm = sol.t_events[0][0]
        M_g = sol.y_events[0][0][1]
    else: # Fallback if the event is not triggered.
        R_cm = sol.t[-1]
        M_g = sol.y[1, -1]

    # Return results in conventional units.
    return R_cm * CM_PER_KM, M_g / M_sun


# --- Main Calculation ---
# A logarithmic scale for central pressures allows us to efficiently
# sample stars from low-mass white dwarfs to high-mass neutron stars.
p_c_bars = np.logspace(-4, 4, 250)
R_tov, M_tov = [], []

print("Calculating TOV M-R curve for a range of central pressures...")
for i, p_bar in enumerate(p_c_bars):
    # It's good practice to wrap the solver in a try-except block
    # to handle cases where integration might fail for extreme parameters.
    try:
        Rt, Mt = integrate_tov(p_bar)
        R_tov.append(Rt)
        M_tov.append(Mt)
    except Exception:
        R_tov.append(np.nan)
        M_tov.append(np.nan)

    if (i + 1) % 50 == 0:
        print(f"  [{i+1:4d}/{len(p_c_bars)}] Calculated star for p_c/eps0 = {p_bar:.4f}")

# Convert lists to numpy arrays for easier analysis.
R_tov = np.array(R_tov)
M_tov = np.array(M_tov)

# --- Find and Report the Maximum Mass Configuration ---
# This is the "TOV Limit" for this simplified EoS.
idx_max = np.nanargmax(M_tov)
R_max = R_tov[idx_max]
M_max = M_tov[idx_max]

print("\n--- Maximum Mass Stable Neutron Star (Fermi Gas EoS) ---")
print(f"  M_max = {M_max:.4f} M_sun")
print(f"  Radius at M_max = {R_max:.4f} km")
print("----------------------------------------------------------")

# --- Plotting the Results ---
plt.style.use('seaborn-v0_8-talk')
plt.figure(figsize=(10, 8))

# Plot the main M-R curve.
plt.plot(R_tov, M_tov, 'r-', linewidth=2.5)

# Highlight the maximum mass point on the curve.
plt.plot(R_max, M_max, 'bo', markersize=8, label=f'Maximum Mass ({M_max:.2f} $M_\\odot$)')
plt.annotate(
    f'$M_{{\\rm max}} = {M_max:.2f}\\,M_\\odot$\n$R = {R_max:.2f}$ km',
    xy=(R_max, M_max),
    xytext=(R_max + 0.5, M_max - 0.2),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=14
)

# Configure plot aesthetics.
plt.xlabel(r'Radius $R$ (km)', fontsize=16)
plt.ylabel(r'Mass $M$ ($M_{\odot}$)', fontsize=16)
plt.title(r'Neutron Star Mass-Radius Relation (Fermi Gas EoS)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14)
plt.xlim(left=5)
plt.ylim(bottom=0)
plt.tight_layout()

# Save figure with high resolution.
plt.savefig('tov_mr_curve_fermi_gas.png', dpi=300)
plt.show()