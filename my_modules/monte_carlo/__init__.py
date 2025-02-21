import sqlite3

from .monte_carlo import Photon, OpticalMedium, System

# Setup default database and MCLUT version
conn = sqlite3.connect('databases/hsdf.db')
c = conn.cursor()
c.execute("SELECT max(id) FROM mclut_simulations")
simulation_id = c.fetchone()[0]

# def lookup(mu_s, mu_a, g, depth, conn=conn, simulation_id=None, force_simulate=True):
#     # Look in the db for closest matching params
#     if simulation_id is None:
#         # If simulation_id is NONE, default to most recent, and then search whole db
#
#     # If params are not an exact match but are in the bounds, interpolate
#
#
#     # If query is out of current bounds or if force simulate is on and params are not an exact match, simulate it
#     elif:



def simulate(system, n=50000, photon=None):
    T, R, A = 3 * [0]
    for i in range(n):
        photon = Photon(650, system=system) if photon is None else photon
        photon.simulate()
        T += photon.T
        R += photon.R
        A += photon.A
    return T / n, R / n, A / n

