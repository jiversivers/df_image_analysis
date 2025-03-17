import os
import sqlite3

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .monte_carlo import Photon, OpticalMedium, System, Illumination, Detector
from my_modules.monte_carlo.hardware import WD

# Setup default database and MCLUT version
conn = sqlite3.connect(f'databases{os.sep}hsdfm_data.db')
c = conn.cursor()
c.execute("SELECT max(id) FROM mclut_simulations")
simulation_id = c.fetchone()[0]


def lookup(mu_s, mu_a, g, depth,
           conn=conn, simulation_id=None, save_sim=True, force_simulate=False, skip_simulation=False):
    assert not (force_simulate and skip_simulation), ('It is ambiguous to set both skip_simulation and force_simulate '
                                                      'to TRUE.')

    # If simulation_id is NONE, default to most recent
    if simulation_id is None:
        c.execute("SELECT * FROM mclut_simulations ORDER BY id DESC LIMIT 1")
        simulation_id = c.fetchone()[0]

    # Fetch available parameters within the simulation_id
    c.execute("""
            SELECT mu_s, mu_a, g, depth, reflectance 
            FROM mclut
            WHERE simulation_id = ?
        """, (simulation_id,))

    rows = c.fetchall()

    # Check for exact matches
    exact_match = None
    for row in rows:
        if row[0] == mu_s and row[1] == mu_a and row[2] == g and row[3] == depth:
            exact_match = row
            break

    if exact_match:
        return exact_match[-1]  # Return the result value for the exact match

    # Removed forced simulations from interpolation data
    c.execute("""
            SELECT mu_s, mu_a, g, depth, reflectance 
            FROM mclut
            WHERE simulation_id = ? AND forced = False
        """, (simulation_id,))
    rows = c.fetchall()

    # If no exact match, check if parameters are within the bounds for interpolation
    mu_s_vals, mu_a_vals, g_vals, depth_vals, ref_vals = zip(*rows)

    # Find nearest bounds for interpolation. Here we assume cubic spline interpolation for all parameters.

    if not force_simulate:
        # Get data into a regular grid
        x = np.unique(mu_s_vals)
        y = np.unique(mu_a_vals)
        z = np.asarray(ref_vals).reshape(len(x), len(y))

        # Interpolation
        spline = RegularGridInterpolator((x, y), z, method='cubic', bounds_error=False, fill_value=None)
        interpolated_ref = spline((mu_s, mu_a))
        return interpolated_ref.item()

    # If query is out of bounds or if force_simulate is True, simulate the result
    elif not skip_simulation:
        print(f'Simulating for mu_s={mu_s}, mu_a={mu_a}, g={g}, depth={depth}')
        # Get matching simulation parameters
        c.execute("""SELECT photon_count, water_n, water_mu_s, water_mu_a, tissue_n, surroundings_n, recursive
            FROM mclut_simulations WHERE id = ?""", (simulation_id,))
        params = c.fetchone()

        # Re-create system
        di_water = OpticalMedium(n=params[1], mu_s=params[2], mu_a=params[3], g=0, name='di_water')
        tissue = OpticalMedium(n=params[4], mu_s=mu_s, mu_a=mu_a, g=g, name='tissue')
        system = System(di_water, WD, tissue, depth, surrounding_n=params[5])

        T, R, A = simulate(system, n=params[0], recurse=params[6])

        # Insert the new simulation into the database
        if save_sim:
            c.execute("""
                    INSERT INTO mclut (simulation_id, mu_s, mu_a, g, depth, transmission, reflectance, absorption, forced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (simulation_id, mu_s, mu_a, g, depth, T, R, A, True))
            conn.commit()

        return R


def simulate(system, n=50000, photon=None, recurse=True):
    T, R, A = 3 * [0]
    for i in range(n):
        current_photon = Photon(650, system=system, recurse=recurse) if photon is None else photon.copy()
        current_photon.simulate()
        T += current_photon.T
        R += current_photon.R
        A += current_photon.A
    return T / n, R / n, A / n
