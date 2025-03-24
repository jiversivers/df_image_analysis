import os.path
from numbers import Real
from typing import Union, Tuple, Iterable, Any

import pandas as pd

import sqlite3

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from src.monte_carlo import System
import importlib.resources

# Setup default database and MCLUT version
con = sqlite3.connect('~/lut.db')
c = con.cursor()
try:
    c.execute("SELECT max(id) FROM mclut_simulations")
    latest_simulation_id = c.fetchone()[0]
except sqlite3.OperationalError:
    latest_simulation_id = None


def lookup(mu_s: Real,
           mu_a: Real,
           g: Real,
           depth: Real,
           conn: sqlite3.Connection = con,
           simulation_id: int = latest_simulation_id,
           extrapolate: bool = True) -> Real:
    assert simulation_id is not None, IOError(
        'No simulations found. Run generate_lut and save results to lut.db before using lookup.')

    # Fetch available parameters within the simulation_id
    c.execute("""
            SELECT mu_s, mu_a, g, depth, reflectance 
            FROM mclut
            WHERE simulation_id = ?
        """, (simulation_id,))
    rows = c.fetchall()
    assert rows is not None and len(rows) > 0, IOError(f'No simulations found at id {simulation_id}. Run generate_lut '
                                                       f'and save results to lut.db before using lookup. or try a '
                                                       f'different ID.')

    # Check for exact matches
    exact_match = None
    for row in rows:
        if row[0] == mu_s and row[1] == mu_a and row[2] == g and row[3] == depth:
            exact_match = row
            break

    if exact_match:
        return exact_match[-1]  # Return the result value for the exact match

    # If no exact match, check if parameters are within the bounds for interpolation
    mu_s_vals, mu_a_vals, g_vals, depth_vals, ref_vals = zip(*rows)

    # Find nearest bounds for interpolation. Here we assume cubic spline interpolation for all parameters.
    # Get data into a regular grid
    x = np.unique(mu_s_vals)
    y = np.unique(mu_a_vals)
    z = np.asarray(ref_vals).reshape(len(x), len(y))

    # Interpolation/Extrapolate (if True)
    spline = RegularGridInterpolator((x, y), z, method='cubic', bounds_error=~extrapolate, fill_value=None)
    interpolated_ref = spline((mu_s, mu_a))

    return interpolated_ref


with importlib.resources.open_text('photon_canon.data', "hbo2_hb.tsv") as f:
    df = pd.read_csv(f, sep='\t', skiprows=1)
wl, hbo2, dhb = df['lambda'], df['hbo2'], df['hb']
wl = np.array([float(w) for w in wl[1:]])
hbo2 = np.array([float(h) for h in hbo2[1:]])
dhb = np.array([float(h) for h in dhb[1:]])
eps = np.stack((hbo2, dhb))

tHb = 4
tHb /= 64500  # molar mass of hemoglobin
sO2 = 0.98


def calculate_mus(a: Real = 1,
                  b: Real = 1,
                  ci: Union[Real, Iterable[Real]] = (tHb * sO2, tHb * (1 - sO2)),
                  epsilons: Union[Iterable[Real], Iterable[Iterable[Real]]] = eps,
                  wavelength: Union[Real, Iterable[Real]] = wl,
                  wavelength0: Real = 650,
                  force_feasible: bool = True) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
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
                assert np.all(np.array([c >= 0 for c in ci])), AssertionError(msg)
            elif isinstance(ci, (int, float)):
                assert ci >= 0, AssertionError(msg)

        # Check that wavelengths and epsilons match up
        msg = (f'A spectrum of molar absorptivity must be included with each spectrum. '
               f'You gave {len(wavelength)} wavelengths but molar absorptivity had {len(epsilons[0])} elements.')
        # Either each element of the epsilons has its own element for the wavelengths
        if isinstance(epsilons[0], (list, tuple, np.ndarray)):
            assert np.all(np.array([len(e) == len(wavelength) for e in epsilons])), AssertionError(msg)
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


def simulate(system: "System", n: int, **kwargs) -> Tuple[float, float, float]:
    photons = system.beam(n=n, **kwargs)
    photons.simulate()
    return photons.T, photons.R, photons.A


def sample_spectrum(wavelengths: Iterable[Real],
                    spectrum: Iterable[Real]):
    wavelengths = np.asarray(wavelengths)
    spectrum = np.asarray(spectrum)

    # Normalize PDF
    spectrum /= np.sum(spectrum)

    # Compute CDF
    cdf = np.cumsum(spectrum)

    # Take random sample
    i = np.random.uniform(0, 1)

    # Interpolate value of sample from CDF
    return np.interp(i, cdf, wavelengths)


def add_metadata(conn=con) -> int:
    # Parse to get metadata

    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut_simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        photon_count INTEGER NOT NULL,
        recursive BOOLEAN DEFAULT FALSE,
        detector BOOLEAN DEFAULT FALSE,
        detector_description TEXT DEFAULT ''
        )
        """)

    c.execute(f"""
    INSERT INTO mclut_simulations (
    photon_count, recursive, detector, detector_description
    ) VALUES (?, ?, ?, ?)""", (
        n, recurse, detector is not None, detector.desc if detector is not None else ''
    ))
    conn.commit()

    return c.lastrowid
