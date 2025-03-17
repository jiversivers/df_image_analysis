import os
import sqlite3

try:
    import cupy as np
    from cupy import iterable

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np
    from numpy import iterable

from scipy.integrate import dblquad

from my_modules.monte_carlo.hardware import ID, THETA, darkfield_footprint

conn = sqlite3.connect(f'databases{os.sep}hsdfm_data.db')
c = conn.cursor()
c.execute(f"SELECT * FROM hb_spectra")
wl, hbo2, dhb = zip(*c.fetchall())
# tHb = 4 and sO2 = 0.98
tHb = 4
tHb /= 64500  # molar mass of hemoglobin
sO2 = 0.98


class Model:
    def __init__(self, function):
        self.function = function

    def fit(self, initials, method, threshold, max_iter):
        pass

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def fresnel_reflection(n1, n2, theta_i):
    # Calculate directional cosines
    mu_z_i = np.cos(theta_i)
    mu_z_t = np.cos(np.arcsin(n1 / n2 * np.sin(theta_i)))

    # Calculate reflection
    Rs = np.abs(((n1 * mu_z_i) - (n2 * mu_z_t)) / ((n1 * mu_z_i) + (n2 * mu_z_t))) ** 2
    Rp = np.abs(((n2 * mu_z_t) - (n1 * mu_z_i)) / ((n1 * mu_z_i) + (n2 * mu_z_t))) ** 2
    return 0.5 * (Rs + Rp)

def calculate_mus(a=1,
                  b=1,
                  ci=(tHb * sO2, tHb * (1 - sO2)),
                  epsilons=(hbo2, dhb),
                  wavelength=wl, wavelength0=650,
                  force_feasible=True):
    # Check cs and epsilons match up
    msg = ('One alpha must be included for all species, but you gave {} ci and {} spectra. '
           'In the case of only two species, the second alpha may be omitted')
    try:
        # Simple 1 to 1 ratio of multiple in list-likes
        if isinstance(ci, (list, tuple, np.ndarray)):
            assert len(ci) == len(epsilons), AssertionError(msg.format(len(ci), len(wavelength)))
        # or 1 ci and either a single list-like OR a one element list-like where that element is list-like
        elif isinstance(ci, (int, float)):
            if isinstance(epsilons[0], (list, tuple, np.ndarray)):
                assert len(epsilons) == 1, AssertionError(msg.format(1, len(epsilons)))

        # Check cs make sense
        if force_feasible:
            msg = 'Concentrations cannot be negative'
            if isinstance(ci, (list, tuple, np.ndarray)):
                assert np.all([c >= 0 for c in ci]), AssertionError(msg)
            elif isinstance(ci, (int, float)):
                assert ci >= 0, AssertionError(msg)

        # Check that wavelengths and epsilons match up
        msg = (f'A spectrum of molar absorptivity must be included with each spectrum. '
               f'You gave {len(wavelength)} wavelengths but molar absorptivity had {len(epsilons[0])} elements.')
        # Either each element of the epsilons has its own element for the wavelengths
        if isinstance(epsilons[0], (list, tuple, np.ndarray)):
            assert np.all([len(e) == len(wavelength) for e in epsilons]), AssertionError(msg)
        # Or there is only one species, and it has its own elements for all wavelengths
        elif isinstance(epsilons[0], (int, float)):
            assert len(epsilons) == len(wavelength), AssertionError(msg)

    except AssertionError as e:
        raise ValueError(e)

    wavelength = np.asarray(wavelength)  # Wavelengths of measurements (nm)
    mu_s = a * (wavelength / wavelength0) ** -b  # Reduced scattering coefficient, cm^-1

    # Unpack list of spectra (if it is a list)
    if isinstance(epsilons[0], (tuple, list, np.ndarray)):
        epsilons = np.asarray([np.asarray(spectrum) for spectrum in epsilons])  # Molar absorptivity (L/(mol cm))
    else:
        epsilons = np.asarray(epsilons)

    # Reshape concentrations (if multiple)
    if isinstance(ci, (list, tuple, np.ndarray)):
        ci = np.asarray(ci)
        ci = ci.reshape(-1, 1)

    mu_a = np.log(10) * np.sum(ci * epsilons, axis=0)  # Absorption coefficient, cm^-1
    return mu_s, mu_a, wl


'''
Diffusion approximation from literature.
- Lihon V. Wang and Hsin-I Wu. Biomedical Optics: Principles and Imaging. p.106-143. John Wiley & Sons, Inc. 2007
- Thomas J Farrell, Machael S Patterson, and Brian Wilson. A diffusion theory model of spatially resolved, steady-state 
diffuse reflectance for the noninvasive determination of tissue optical properties in vivo. (1992). Med Phys 19:4 
p.879-888. doi: 10.1118/1.596777
'''

# Get some default values
mu_s, mu_a, wl = calculate_mus()

# This will be the base and simplest form to be built up from. This assumes point, normal incidence at thr origin and
# returns the reflectance at a radial distance, r, from the origin.
def diffusion_approximation(mu_s=mu_s, mu_a=mu_a,
                            r=0, n_tissue=1.33, n_collection=1.33, g=0.9):

    # Reduce the scattering coefficient
    mu_s *= (1 - g)

    a_correction = 1 - fresnel_reflection(n_collection, n_tissue, theta)

    # Optical properties (and derivatives)
    mu_t = mu_a + mu_s  # Total interaction coefficient, cm^-1
    a = mu_s / mu_t  # In text (intro while discussing Patterson et al.)
    D = 1 / (3 * mu_t)  # Diffusion constant; Eqn. 3
    mu_eff = np.sqrt(3 * mu_a / D)

    z = 1 / mu_t
    zb = -2 * D

    rho1 = np.sqrt(r ** 2 + z ** 2)
    rho2 = np.sqrt(r ** 2 + (-z - 2 * zb) ** 2)

    # Green's Function for Diffuse Reflectance; Eqn. 15
    rd = (
            (a / (4 * np.pi)) *
            ((z * (1 + (mu_eff * rho1)) * np.exp(-mu_eff * rho1)) /
             (rho1 ** 3)) +
            (((z + (4 * D)) * (1 + (mu_eff * rho2)) * np.exp(-mu_eff * rho2)) /
             (rho2 ** 3))
    )

    return rd


'''
The base diffusion approximation must be extended to account for non-normal incidence and a finite source. To account 
for the finite source, the reflectance is integrated across the beam.
'''
def darkfield_reflectance(r_bounds):

    pass

# Get some default values
mu_s, mu_a, wl = calculate_mus()
def integrated_reflectance(mu_s=mu_s, mu_a=mu_a, r_bounds=(0, ID), theta_bounds=THETA, beam_function=darkfield_footprint,
                           n_tissue=1.33, n_collection=1.33, g=0.9):
    if not iterable(r_bounds):
        r_bounds = [r_bounds, r_bounds]
    if not iterable(theta_bounds):
        theta_bounds = [theta_bounds, theta_bounds]

    A = beam_function(inner=r_bounds[0], outer=r_bounds[1], theta_min=theta_bounds[0], theta_max=theta_bounds[1])

    def integrand(r, theta):
        return (diffusion_approximation(mu_s=mu_s, mu_a=mu_a, theta=theta, r=r,
                                        n_tissue=n_tissue, n_collection=n_collection, g=g)
                * beam_function(inner=r_bounds[0], outer=r_bounds[1],
                                theta_min=theta_bounds[0], theta_max=theta_bounds[1]))

    integral, _ = dblquad(integrand, theta_bounds[0], theta_bounds[1], lambda _: r_bounds[0], lambda _: r_bounds[1])



# TODO: Add LUT load and get value and update get_optical_properties to use it. Make it match the same inputs and
#  returns style of diffusion_approximation

def mclut(mus, mua, rho=2.25, n_tissue=1.4, n_collection=1):
    pass


# TR = p50 from Boltzmann Function
# R is the average rate-limited distance from center of capillaries (dist when NADH int = 50% max)
# r is capillary radius measured from the HSDF image vascular map
# T0 is the oxygen tension within the capillary, as measured from the mclut or diffusion approximation
def krogh_cylinder(T0, TR, R, r, x):
    p_K = (T0 - TR) / ((1 / 2) * (R ** 2) * np.log(R / r) - ((R ** 2 - r ** 2) / 4))
    Tx = T0 - p_K * ((1 / 2) * (R ** 2) * np.log(R / r) - ((R ** 2 - r ** 2) / 4))
    return Tx


# po2 is the oxygen tension within the capillary, as measured from the mclut or diffusion approximation
def boltzmann_function(a1, a2, pO2, p50, dx):
    nadph_per = a2 + ((a1 + a2) / (1 + np.exp((pO2 - p50) ** dx)))
    return nadph_per
