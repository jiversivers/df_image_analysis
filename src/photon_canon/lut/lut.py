import itertools
import sqlite3
from typing import Optional

from numpy.typing import NDArray
from tqdm import tqdm

from ..optics import System, Medium
from ..optics import Photon

from ..lut.utils import add_metadata, add_system_data, add_simulation_result


def generate_lut(
        system: System,
        variable: Medium,
        arrays: dict[str, NDArray],
        photon: Photon,
        pbar: bool = False,
        output: Optional[str] = None,
        **kwargs
) -> int:
    """
    Simulates photon transport for different optical properties to generate a lookup table (LUT).

    This function iterates over the input arrays, modifying the optical properties of `variable`
    (a `Medium` object) accordingly. A set of `Photon` objects is then simulated within the `system`.

    :param system: The optical system in which the photons are simulated.
    :type system: System
    :param variable: The medium whose properties are varied in the LUT generation.
    :type variable: Medium
    :param arrays: Dictionary mapping property names to NumPy arrays containing values to iterate over.
    :type arrays: dict[str, np.ndarray]
    :param photon: A reference `Photon` object used to initialize the simulated photons.
    :type photon: Photon
    :param pbar: Whether to show a progress bar.
    :type pbar: bool
    :param kwargs: Additional optional parameters for the simulation.
    :return: The ID of the generated LUT of simulation.
    :rtype: int
    """

    # Add LUT metadata to db
    simulation_id = add_metadata(n=photon.batch_size, recursive=photon.recursive, detector=system.detector)
    add_system_data(simulation_id, system)

    # Iterate through all permutations of iterables
    iterable = tqdm(itertools.product(*arrays.values()),
                    total=sum([len(arr) for arr in arrays.values()])) if pbar else itertools.product(*arrays.values())
    copy_of_photon = photon.copy()
    for values in iterable:
        keys_values = dict(zip(arrays.keys(), values))
        if pbar:
            iterable.set_description(
                f' - '.join(f"{k}: {v}" for k, v in keys_values.items())
            )
        variable.set(**keys_values)

        # Reset and simulate
        photon = copy_of_photon.copy()
        if system.detector is not None:
            system.detector.reset()
        photon.simulate()

        # Get the output
        if system.detector is not None:
            target = system.detector.n_detected
        elif output is not None:
            target = getattr(photon, output, None)
        else:
            target = getattr(photon, 'R', None)

        # Update LUT
        for bound, layer in system.stack:
            if layer == variable:
                depth = bound[1] = bound[0]
                break
        add_simulation_result(simulation_id, variable.mu_s, variable.mu_a, variable.g, depth, )
