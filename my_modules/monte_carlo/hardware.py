import numpy as np
from numpy import iterable

# Hardware specs in cm
ID = 0.16443276801785274
OD = 0.3205672319821467
WD = 0.2

THETA = np.arctan(-OD / WD)  # rad
NA = 1.0


def ring_pattern(r_bounds, angle_bounds):
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

    def sampler():
        # Sample angle and radius for starting location
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2))

        # Create ring
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        location = (x, y, 0)

        # Sample injection angles directional cosines
        theta = np.random.uniform(angle_min, angle_max)

        # Compute directional cosines
        mu_x = np.sin(theta) * np.cos(phi)
        mu_y = np.sin(theta) * np.sin(phi)
        mu_z = np.cos(theta)
        directional_cosines = (mu_x, mu_y, mu_z)

        return location, directional_cosines

    return sampler


def cone_of_acceptance(r, na=NA, n=1.33):
    def acceptor(x, y, mu_z=None):
        if mu_z is not None:
            theta_max = np.arcsin(na / n)
            mu_z_max = np.cos(theta_max)
            too_steep = mu_z > mu_z_max
        else:
            too_steep = False
        r_test = np.sqrt(x ** 2 + y ** 2)
        outside = r_test > r
        return not (too_steep or outside)

    return acceptor

