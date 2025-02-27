import sqlite3
from scipy.interpolate import CubicSpline
from .monte_carlo import Photon, OpticalMedium, System

# Setup default database and MCLUT version
conn = sqlite3.connect('databases/hsdfm_data.db')
c = conn.cursor()
c.execute("SELECT max(id) FROM mclut_simulations")
simulation_id = c.fetchone()[0]


def lookup(mu_s, mu_a, g, depth, conn=conn, simulation_id=None, force_simulate=True, recurse=True):
    if simulation_id is None:
        # If simulation_id is NONE, default to most recent, and then search the whole db
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

    # If no exact match, check if parameters are within the bounds for interpolation
    mu_s_vals, mu_a_vals, g_vals, depth_vals, ref_vals = zip(*rows)

    # Find nearest bounds for interpolation. Here we assume cubic spline interpolation for all parameters.
    if not force_simulate and (min(mu_s_vals) <= mu_s <= max(mu_s_vals) and
                               min(mu_a_vals) <= mu_a <= max(mu_a_vals) and
                               min(g_vals) <= g <= max(g_vals) and
                               min(depth_vals) <= depth <= max(depth_vals)):
        # Interpolation setup
        spline = CubicSpline(mu_s_vals, ref_vals)
        interpolated_value = spline(mu_s)
        return interpolated_value

    # If query is out of bounds or if force_simulate is True, simulate the result
    else:
        # Here, simulate a new result. This could be a complex function or simulation based on mu_s, mu_a, g, depth
        T, R, A = simulate(mu_s, mu_a, g, depth, recurse=recurse)

        # Insert the new simulation into the database
        c.execute("""
            INSERT INTO mclut (simulation_id, mu_s, mu_a, g, depth, transmission, reflectance, absorption, forced)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (simulation_id, mu_s, mu_a, g, depth, T, R, A, True))
        conn.commit()

        return R


def simulate(system, n=50000, photon=None, recurse=True):
    T, R, A = 3 * [0]
    for i in range(n):
        photon = Photon(650, system=system, recurse=recurse) if photon is None else photon
        photon.simulate()
        T += photon.T
        R += photon.R
        A += photon.A
    return T / n, R / n, A / n
