"""
Solves the Tolman-Oppenheimer-Volkoff (TOV) equations to calculate the
mass-radius relation for neutron stars using the realistic, tabulated
SLy Equation of State for dense nuclear matter.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from astropy import constants as const

# --- Physical Constants (CGS units) ---
# Using astropy for precise, standard values.
G = const.G.cgs.value        # Gravitational constant (cm^3/g/s^2)
c = const.c.cgs.value        # Speed of light (cm/s)
M_sun = const.M_sun.cgs.value  # Solar mass (g)
CM_PER_KM = 1e5              # Conversion factor from cm to km

# --- Load and Interpolate the Equation of State (EoS) ---
# The SLy EoS is a realistic model for cold, dense nuclear matter,
# provided as a table of thermodynamic properties.
sly_data = np.genfromtxt("SLy.txt", delimiter="  ")
mass_density = sly_data[:, 2]    # Mass-energy density (g/cm^3)
pressure = sly_data[:, 3]        # Pressure (dyne/cm^2)

# To use the EoS in the differential equation solver, we need a continuous
# function. CubicSpline creates a smooth interpolation from the discrete table points.
pressure_from_density = CubicSpline(mass_density, pressure)
density_from_pressure = CubicSpline(pressure, mass_density)


def tov_equations(r, y):
    """
    Defines the Tolman-Oppenheimer-Volkoff (TOV) equations of stellar structure.

    Args:
        r (float): The current radius in cm.
        y (list): The state vector [enclosed_mass, pressure].

    Returns:
        list: The derivatives [dM/dr, dP/dr].
    """
    mass, pressure_val = y

    # Stop integration if pressure becomes non-physical.
    if pressure_val <= 0.0:
        return [0.0, 0.0]

    # The EoS provides the crucial link between pressure and density.
    rho = density_from_pressure(pressure_val)

    # --- Equation 1: Mass continuity ---
    dm_dr = 4.0 * np.pi * r**2 * rho

    # --- Equation 2: Hydrostatic equilibrium in GR ---
    # This is the TOV equation, which includes several relativistic corrections.
    num = G * mass * rho / r**2
    fac1 = 1.0 + pressure_val / (rho * c**2)                # Relativistic correction for inertia
    fac2 = 1.0 + 4.0 * np.pi * r**3 * pressure_val / (mass * c**2)  # Correction for pressure as a source of gravity
    fac3 = 1.0 - 2.0 * G * mass / (r * c**2)              # Spacetime curvature (redshift) correction

    # The full GR pressure gradient.
    dP_dr = -num * fac1 * fac2 / fac3

    return [dm_dr, dP_dr]


def solve_for_one_star(central_density, r_initial=1.0, r_max=2e6):
    """
    Integrates the TOV equations for a single star with a given central density.

    Args:
        central_density (float): The density at the star's center (g/cm^3).
        r_initial (float): A small, non-zero radius to start integration and avoid singularity.
        r_max (float): Maximum radius to integrate out to.

    Returns:
        tuple: (Total Radius in cm, Total Mass in g).
    """
    # Determine the central pressure from the central density using the EoS.
    central_pressure = pressure_from_density(central_density)

    # Initial conditions for the integration at r = r_initial.
    initial_mass = (4.0/3.0) * np.pi * r_initial**3 * central_density
    y0 = [initial_mass, central_pressure]

    # --- Define a termination event for the solver ---
    # The star's surface is defined as the radius where pressure drops to zero.
    def pressure_is_zero(r, y):
        return y[1]
    pressure_is_zero.terminal = True  # Stop the integration when this event occurs.
    pressure_is_zero.direction = -1   # Trigger only for a decreasing pressure.

    # Integrate the TOV equations from the center outwards.
    solution = solve_ivp(
        tov_equations, [r_initial, r_max], y0,
        method='RK45', events=pressure_is_zero, rtol=1e-6
    )

    # Extract the final radius and mass from the event.
    if solution.t_events[0].size > 0:
        radius = solution.t_events[0][0]
        mass = solution.y_events[0][0][0]
    else: # Fallback if the solver finishes before finding the surface.
        radius = solution.t[-1]
        mass = solution.y[0, -1]

    return radius, mass


# --- Main Calculation Loop ---
# To generate a mass-radius curve, we solve for a sequence of stars,
# each with a different central density.
central_densities = np.linspace(2.5e14, 5e15, 200)

# Store the results for each star.
radius_array = np.zeros_like(central_densities)
mass_array = np.zeros_like(central_densities)

print("Solving TOV equations for a range of central densities (SLy EoS)...")
for i, rho_c in enumerate(central_densities):
    radius_cm, mass_g = solve_for_one_star(rho_c)
    radius_array[i] = radius_cm
    mass_array[i] = mass_g
    if (i + 1) % 50 == 0:
        print(f"  ... {i+1}/{len(central_densities)} stars calculated.")

# Convert final results to standard astrophysical units for plotting.
R_tov_km = radius_array / CM_PER_KM
M_tov_Msun = mass_array / M_sun

# --- Find and Report the Maximum Mass Configuration ---
# The peak of the M-R curve is the TOV limit: the maximum mass a
# non-rotating star can support before collapsing to a black hole.
idx_max = np.nanargmax(M_tov_Msun)
R_max_km = R_tov_km[idx_max]
M_max_Msun = M_tov_Msun[idx_max]

print("\n--- Maximum Mass Stable Neutron Star (SLy EoS) ---")
print(f"  M_max = {M_max_Msun:.4f} M_sun")
print(f"  Radius at M_max = {R_max_km:.4f} km")
print("--------------------------------------------------")

# --- Plotting the Mass-Radius Curve ---
plt.style.use('seaborn-v0_8-talk')
plt.figure(figsize=(10, 8))

# Plot the main curve.
plt.plot(R_tov_km, M_tov_Msun, 'r-', linewidth=2.5)

# Highlight the maximum mass point.
plt.plot(R_max_km, M_max_Msun, 'bo', markersize=8, label=f'Maximum Mass ({M_max_Msun:.2f} $M_\\odot$)')
plt.annotate(
    f'$M_{{\\rm max}} = {M_max_Msun:.2f}\\,M_\\odot$\n$R = {R_max_km:.2f}$ km',
    xy=(R_max_km, M_max_Msun), xytext=(R_max_km + 1, M_max_Msun - 0.4),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=14
)

# Configure plot aesthetics.
plt.xlabel(r'Radius $R$ (km)', fontsize=16)
plt.ylabel(r'Mass $M$ ($M_{\odot}$)', fontsize=16)
plt.title(r'Neutron Star Mass-Radius Relation (SLy EoS)', fontsize=18)
plt.xlim(8, 16)
plt.ylim(0, 2.5)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure with high resolution.
plt.savefig('tov_mr_curve_SLy.png', dpi=300, bbox_inches='tight')
plt.show()