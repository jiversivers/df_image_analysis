import sqlite3

import numpy as np

conn = sqlite3.connect(r'C:\Users\jdivers\PycharmProjects\df_image_analysis\databases\hsdfm_data.db')
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


# TODO: Determine how to handle RHO for multiple and non-normal beams.
mu_s, mu_a, wl = calculate_mus()


def diffusion_approximation(mu_s=mu_s, mu_a=mu_a,
                            rho=2.25e-3, n_tissue=1.4, n_collection=1, g=None):
    # Optical properties (and derivatives)
    mu_t = mu_a + mu_s  # Total interaction coefficient, cm^-1
    mu_eff = np.sqrt(3 * mu_a * mu_t)  # In text (intro while discussing Patterson et al.)
    albedo = mu_s / mu_t  # In text (intro while discussing Patterson et al.)
    D = 1 / (3 * mu_t)  # Diffusion constant; Eqn. 3

    # System Properties
    n_rel = n_tissue / n_collection  # Relative refractive index; in text (preceding eqn. 7)
    r_d = 1.440 * (n_rel ** (-2)) + 0.710 * (n_rel ** (-1)) + 0.668 + 0.0636 * n_rel  # Eqn. 9
    A = (1 + r_d) / (1 - r_d)  # Eqn. 8

    z_b = 2 * A * D  # In text (fluence zero point; boundary condition explanation following Eqn. 9)
    z = 0  # In text (following solution explanation of eqn. 15)
    z_0 = 1 / mu_t  # In text (preamble to Eqn. 18)

    r1 = np.sqrt(((z - z_0) ** 2 + rho ** 2))  # Green's function for fluence parameter; Eqn. 11
    r2 = np.sqrt(((z + z_0 + 2 * z_b) ** 2 + rho ** 2))  # Green's function solved for a point source; Eqn. 13

    # Green's Function for Diffuse Reflectance; Eqn. 15
    R = (albedo / (4 * np.pi)) * (z_0 * (mu_eff + (1 / r1)) * (np.exp(-mu_eff * r1) / (r1 ** 2)) +
                                  ((z_0 + 2 * z_b) * (mu_eff + (1 / r2)) * (np.exp(-mu_eff * r2) / (r2 ** 2))))

    return R


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
