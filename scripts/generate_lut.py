import itertools
import sqlite3

from src import Medium, System, Detector, Illumination, hardware

from tqdm import tqdm

try:
    import cupy as np

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np

    np.is_available = lambda: False  # Mock the `is_available` method for consistency

print(f"Using {np.__name__}")


def main():
    # Init parameter sets
    mu_s_array = np.arange(0, 101, 1)
    mu_a_array = np.arange(1, 102, 1)
    g_array = [0.9]
    d = float('inf')
    n = 50000
    tissue_n = 1.33
    surroundings_n = 1
    recurse = False
    wl0 = 700

    # Make water medium
    di_water = Medium(n=1.33, mu_s=0, mu_a=0, g=0, desc='di water')
    glass = Medium(n=1.523, mu_s=0, mu_a=0, g=0, desc='glass')

    # Start the system
    sampler = hardware.ring_pattern((hardware.ID, hardware.OD), hardware.THETA)
    led = Illumination(pattern=sampler)
    detector = Detector(hardware.cone_of_acceptance(hardware.ID), desc='darkfield inner cone')
    system = System(di_water, 0.2,  # 1mm
                    glass, 0.017,  # 0.17mm
                    surrounding_n=surroundings_n,
                    illuminator=led,
                    detector=(detector, 0))

    # Simulate
    conn = sqlite3.connect('data/databases/lut.db')
    c = conn.cursor()

    # Table of metadata
    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut_simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        photon_count INTEGER NOT NULL,
        recursive BOOLEAN DEFAULT FALSE,
        detector BOOLEAN DEFAULT FALSE,
        detector_description TEXT DEFAULT ''
        )
        """)

    # Table of detailed system data
    c.execute("""
    CREATE TABLE IF NOT EXISTS fixed_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stack_order INTEGER NOT NULL,
    layer TEXT NOT NULL,
    mu_s REAL NOT NULL,
    mu_a REAL NOT NULL,
    g REAL NOT NULL,
    thickness REAL NOT NULL,
    ref_wavelength REAL NOT NULL,
    simulation_id INTEGER NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES mclut_simulations(id)
    )
    """)

    # Table of results
    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mu_s REAL NOT NULL,
        mu_a REAL NOT NULL,
        g REAL NOT NULL,
        depth REAL NOT NULL,
        reflectance REAL NOT NULL,
        simulation_id INTEGER NOT NULL,
        FOREIGN KEY (simulation_id) REFERENCES mclut_simulations(id)
        )
        """)

    # Insert simulation metadata
    c.execute(f"""
    INSERT INTO mclut_simulations (
    photon_count, recursive, detector, detector_description
    ) VALUES (?, ?, ?, ?)""", (
        n, recurse, detector is not None, detector.desc if detector is not None else ''
    ))
    conn.commit()
    simulation_id = c.lastrowid

    # Add fixed layer details to table
    fixed_layers = []
    for i, (bound, layer) in enumerate(system.stack.items()):
        fixed_layers.append((
            i, layer.desc, layer.mu_s_at(wl0), layer.mu_a_at(wl0), layer.g, bound[1] - bound[0], wl0, simulation_id
        ))
    c.executemany(f"""
    INSERT INTO fixed_layers (
        stack_order, layer, mu_s, mu_a, g, thickness, ref_wavelength, simulation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", fixed_layers)

    # Iterate through input params
    tissue = Medium(n=tissue_n, mu_s=0, mu_a=0, g=1, desc='tissue')  # Placeholder to update at iteration
    system.add(tissue, d)
    detected_R = None
    for (mu_s, mu_a, g) in (pbar := tqdm(itertools.product(mu_s_array, mu_a_array, g_array),
                                         total=len(mu_s_array) * len(mu_a_array) * len(g_array))):
        pbar.set_description(
            f'{'' if detected_R is None else f'Prev. R: {detected_R} |'} '
            f'Current params: mu_s={mu_s}, mu_a={mu_a}, g={g}...')

        # Update the system
        tissue.set(mu_s=mu_s, mu_a=mu_a, g=g)
        # Reset counter
        detector.reset()

        photon = system.beam(n=n, recurse=recurse, tir_limit=100, russian_roulette_constant=20)
        photon.simulate()
        detected_R = detector.n_detected / (n - (photon.R - detector.n_detected))

        # Add results to db
        c.execute(f"""
                    INSERT INTO mclut (
                    mu_s, mu_a, g, depth, reflectance, simulation_id
                    ) VALUES (?, ?, ?, ?, ?, ?)""", (
            float(mu_s), float(mu_a), float(g), float(d), float(detected_R), simulation_id
        ))
    conn.commit()


if __name__ == '__main__':
    main()
