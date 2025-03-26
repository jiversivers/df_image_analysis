import sqlite3
from numbers import Real
from pathlib import Path
from typing import Union, Iterable

import numpy as np
from .. import System

# Set up simulation database
db_dir = Path.home() / ".photon_canon"
db_dir.mkdir(parents=True, exist_ok=True)
db_path = db_dir / "lut.db"
con = sqlite3.connect(db_path)
c = con.cursor()

def add_metadata(n=None, recursive=False, detector=None) -> int:
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
        n, recursive, detector is not None, detector.desc if detector is not None else ''
    ))
    con.commit()

    return c.lastrowid

def add_system_data(simulation_id: int, system: System) -> None:
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

    # Generate fixed layer details for table
    fixed_layers = []
    for i, (bound, layer) in enumerate(system.stack.items()):
        fixed_layers.append((
            int(i),
            layer.desc,
            float(layer.mu_s_at(layer.ref_wavelength)),
            float(layer.mu_a_at(layer.ref_wavelength)),
            float(layer.g),
            float(bound[1] - bound[0]),
            float(layer.ref_wavelengths),
            int(simulation_id)
        ))

    # Add details to table
    c.executemany(f"""
    INSERT INTO fixed_layers (
        stack_order, layer, mu_s, mu_a, g, thickness, ref_wavelength, simulation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", fixed_layers)
    con.commit()


def add_simulation_result(simulation_id: int,
                          mu_s: Union[Real, Iterable[Real]],
                          mu_a: Union[Real, Iterable[Real]],
                          g: Union[Real, Iterable[Real]],
                          depth: Union[Real, Iterable[Real]],
                          output: Union[Real, Iterable[Real]]) -> None:
    # Ensure all are iterable arays
    arrays = [np.array(val) for val in [mu_s, mu_a, g, depth, output, simulation_id]]
    iters = [arr.ndim == 1 for arr in arrays]
    shapes = [arrays[idx].shape for idx, itr in enumerate(iters) if itr]

    # Check that sizes are compatible
    if not np.all(shapes == shapes[0]):
        raise ValueError('Iterables must have the same shape')

    # Expand non-iterables to match and make all floats
    for i, itr in enumerate(iters):
        if not itr:
            arrays[i] == np.repeat(float(arrays[i]), shapes[0][1])
        else:
            arrays[i] = [float(val) for val in arrays[i]]

    # Put into tuples for executemany
    data_tuples = tuple((zip(arrays)))

    # Add table (if not exists)
    c.execute("""
     CREATE TABLE IF NOT EXISTS mclut (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         mu_s REAL NOT NULL,
         mu_a REAL NOT NULL,
         g REAL NOT NULL,
         depth REAL NOT NULL,
         output REAL NOT NULL,
         simulation_id INTEGER NOT NULL,
         FOREIGN KEY (simulation_id) REFERENCES mclut_simulations(id)
         )
         """)

    # Add results to db
    c.executemany(f"""
                INSERT INTO mclut (
                mu_s, mu_a, g, depth, output, simulation_id
                ) VALUES (?, ?, ?, ?, ?, ?)""", data_tuples)

    con.commit()
