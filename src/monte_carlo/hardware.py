from numbers import Real
from typing import Tuple, Union, Callable, Optional, Iterable

try:
    import cupy as np
    from cupy import iterable
    from cupy.typing import NDArray

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np
    from numpy import iterable
    from numpy.typing import NDArray

    np.is_available = lambda: False  # Mock the `is_available` method for consistency

# Hardware specs in cm
ID = 0.16443276801785274
OD = 0.3205672319821467
WD = 0.2

THETA = np.arctan(-OD / WD)  # rad
NA = 1.0

# Create random number generator
rng = np.random.default_rng()


# Formulation for use in analytical modelling.
# NOTE: This gives the beam _at_ a distance WD from where ID/OD were measured, presumably at the medium of measure.
# theta_max serves as a compatibility placeholder, but is not used for darkfield_footprint.
def darkfield_footprint(inner: Real = ID,
                        outer: Real = OD,
                        working_distance: Real = WD,
                        theta_min: Real = THETA) -> Real:
    # Calculate radii' at WD
    inner = inner / 2 + working_distance * np.tan(theta_min)
    outer = outer / 2 + working_distance * np.tan(theta_min)

    # Calculate area of circles and total beam area
    return np.pi * (outer ** 2 - inner ** 2)


# Sampler for use in Monte Carlo modelling
# NOTE: This samples the ring _at_ where ID/OD are measured from, presumably a distance WD above the medium of measure.
def ring_pattern(r_bounds: Union[Real, Tuple[Real, Real]],
                 angle_bounds: Union[Real, Tuple[Real, Real]]) -> Callable:
    if not iterable(r_bounds):
        r_max = r_bounds
        r_min = 0
    elif len(r_bounds) == 1:
        r_max = r_bounds[0]
        r_min = 0
    else:
        r_min, r_max = r_bounds

    if not iterable(angle_bounds):
        angle_min = angle_bounds
        angle_max = angle_bounds
    elif len(angle_bounds) == 1:
        angle_min = angle_bounds[0]
        angle_max = angle_bounds[0]
    else:
        angle_min, angle_max = angle_bounds

    def sampler(n: int = 50000) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Sample angle and radius for starting location
        phi = np.random.uniform(0, 2 * np.pi, n)
        r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2, n))

        # Create ring (2D coordinates)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        location = np.vstack((x, y, np.zeros(n))).T  # Stack x, y, and z=0 into shape (n, 3)

        # Sample injection angles directional cosines
        theta = np.random.uniform(angle_min, angle_max, n)

        # Compute directional cosines
        mu_x = np.sin(theta) * np.cos(phi)
        mu_y = np.sin(theta) * np.sin(phi)
        mu_z = np.cos(theta)
        directional_cosines = np.vstack((mu_x, mu_y, mu_z)).T  # Stack into shape (n, 3)

        return location, directional_cosines

    return sampler


def cone_of_acceptance(r: Real,
                       na: Real = NA,
                       n: Real = 1.33) -> Callable:
    def acceptor(x: Union[Real, Iterable[Real]],
                 y: Union[Real, Iterable[Real]],
                 mu_z: Union[Real, Iterable[Real]] = None) -> NDArray[np.bool_]:
        x = np.array(x)
        y = np.array(y)
        if mu_z is not None:
            theta_max = np.arcsin(na / n)
            mu_z_max = np.cos(theta_max)
            too_steep = np.array(mu_z) > mu_z_max
        else:
            too_steep = False
        r_test = np.sqrt(x ** 2 + y ** 2)
        outside = r_test > r
        return ~too_steep & ~outside

    return acceptor
