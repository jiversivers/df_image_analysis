import sqlite3

import numpy as np
from matplotlib import pyplot as plt


class System:
    def __init__(self, *args, surrounding_n=1):
        """
        Args comes in groups of 2:
            0: the medium
            1: the thickness
        The blocks are constructed top down in the order they are received and with a semi-infinite approximation; i.e.
        the top block starts at inf (typically air or water).
        """
        assert len(args) % 2 == 0, "Arguments must be in groups of 2: medium object and thickness."
        self.surroundings = OpticalMedium(n=surrounding_n, type='surroundings')

        # Stack layers
        interface = 0
        self.stack = {(float('-inf'), interface): self.surroundings}
        self.layer = [self.surroundings]
        boundaries = []
        for i in range(0, len(args), 2):
            self.stack[(interface, interface + args[i + 1])] = args[i]
            self.layer.append(args[i])
            boundaries.append(interface)
            interface += args[i + 1]

        # Last layer finish
        boundaries.append(interface)
        self.boundaries = np.asarray(boundaries)
        self.stack[(interface, float('inf'))] = self.surroundings

    def in_medium(self, location):
        # Return the medium(s) that are at the queried coordinate. If the coordinate is an interface location, the
        # mediums that makeup the interface are returned as a tuple, this includes boundary interfaces being returned
        # with False on the surroundings side.
        z = location[2] if isinstance(location, (tuple, list, np.ndarray)) else location
        for bound, medium in self.stack.items():
            if bound[0] < z < bound[1]:
                in_ = medium
                break
            elif z == bound[0]:
                in_ = (self.in_medium(np.nextafter(z, float('-inf'))),
                       self.in_medium(np.nextafter(z, float('inf')))
                       )
                break
        return in_

    def interface_crossed(self, location1, location2):
        # Get z coords
        z = []
        for location in [location1, location2]:
            z.append(location[2] if isinstance(location, (tuple, list, np.ndarray)) else location)

        for i, loc in enumerate(z):
            if loc in self.boundaries:
                # Bumps the coords towards the other z.
                # When i = 0, coords are bumped forward in the direction of z_dir (toward z[1])
                # When i = 1, coords are bumped backwards of the direction of z_dir (toward z[0])
                z[i] = np.nextafter(loc, z[1 - i])

        # Get direction of vector
        z_dir = np.sign(z[1] - z[0])

        # Sort for easier logic
        z_sorted = np.sort(z)

        # Check if any boundaries fall between the zs
        crossed = [z_sorted[0] < bound < z_sorted[1] for bound in self.boundaries]

        # Determine the distance of each boundary from the start coordinate
        dist_from_start = np.abs(self.boundaries - z[0])
        dist_from_start[[not cross for cross in crossed]] = float('inf')
        plane = self.boundaries[crossed and dist_from_start == np.min(dist_from_start)][0] if not np.all(
            dist_from_start == float('inf')) else False

        # If an interface is crossed, get the two mediums
        interface = []
        if plane:
            interface.append(self.in_medium(np.nextafter(plane, -1 * z_dir * np.finfo(np.float64).max)))
            interface.append(self.in_medium(np.nextafter(plane, z_dir * np.finfo(np.float64).max)))

        return tuple(interface), plane

    def represent_on_axis(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_ylim([-0.1 * self.boundaries[-1], 1.1 *self.boundaries[-1]])
        alpha = 0.0
        for bound, medium in self.stack.items():
            depth = np.diff(bound)
            y_edge = bound[0] - 0.1 * depth
            x_edge = ax.set_xlim(ax.get_xlim())[0] * 0.95
            ax.text(x_edge, y_edge, medium.type, fontsize=12)
            line_x = 100 * np.asarray(ax.get_xlim())
            alpha += 0.2
            ax.fill_between(line_x, bound[0], bound[1],
                            color='gray' if medium.color is None else medium.color,
                            alpha=alpha if medium.color is None else 1)


class Photon:
    def __init__(self, wavelength,
                 system=None,
                 directional_cosines=(0, 0, 1),
                 location_coordinates=(0, 0, 0),
                 weight=1,
                 russian_roulette_constant=20):

        # Init photon state
        self.wavelength = wavelength
        self.system = system
        self.directional_cosines = np.asarray(directional_cosines, dtype=np.float64)
        self.location_coordinates = np.asarray(location_coordinates, dtype=np.float64)
        self.weight = weight
        self.russian_roulette_constant = russian_roulette_constant
        self._medium = None

        # Init trackers
        self.location_history = [self.location_coordinates]
        self.A = 0.0
        self.T = 0.0
        self.R = 0.0

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        if 0 < weight < 0.005:
            self.russian_roulette()
        elif weight <= 0:
            self._weight = 0
        else:
            self._weight = weight

    def russian_roulette(self):
        if np.random.rand() < 1 / self.russian_roulette_constant:
            self._weight = 0
        else:
            self._weight * self.russian_roulette_constant

    @property
    def medium(self):
        self._medium = self.system.in_medium(self.location_coordinates)
        if isinstance(self._medium, (list, tuple)):
            self._medium = self.headed_into
        return self._medium

    @property
    def is_terminated(self):
        return (self.medium == self.system.surroundings or self.weight <= 0.0)

    @property
    def headed_into(self):
        bumped_coords = self.location_coordinates
        medium = []
        while isinstance(medium, (list, tuple)):
            medium = self.system.in_medium(bumped_coords)
            # Increment coordinates smallest possible amount in same direction of the photon
            bumped_coords = np.nextafter(self.location_coordinates,
                                         np.finfo(np.float64).max * np.sign(self.directional_cosines))
        return medium

    def move(self, step=None):
        mu_t = self.medium.mu_t
        step = -np.log(np.random.rand()) / mu_t if step is None else step
        d_loc = step * self.directional_cosines
        new_coords = d_loc + self.location_coordinates

        # If an interface is crossed
        interface, plane = self.system.interface_crossed(self.location_coordinates, new_coords)
        while interface and plane:
            # Update new_coords to the fraction of the step to the interface
            step_frac = (plane - self.location_coordinates[2]) / d_loc[2]
            new_coords = self.location_coordinates + step_frac * d_loc
            self.location_coordinates = new_coords
            self.location_history.append(new_coords)

            # Reflect and refract
            self.reflect(interface)
            self.refract(interface)

            # Move the rest, rescaling the remaining step for the new medium
            step *= (1 - step_frac) * mu_t / self.medium.mu_t
            mu_t = self.medium.mu_t
            d_loc = step * self.directional_cosines
            new_coords = d_loc + self.location_coordinates
            interface, plane = self.system.interface_crossed(self.location_coordinates, new_coords)
        # Final non-crossing move
        self.location_coordinates = new_coords
        self.location_history.append(new_coords)

        if self.location_coordinates[2] < self.system.boundaries[0]:
            self.R += self.weight
            self.weight = 0
        elif self.location_coordinates[2] > self.system.boundaries[-1]:
            self.T += self.weight
            self.weight = 0

    def refract(self, interface):
        sin_theta_t = (interface[0].n / interface[1].n) * np.sqrt(1 - self.directional_cosines[2] ** 2)

        # Critical angle reflection
        if sin_theta_t > 1:
            self.directional_cosines[2] = -self.directional_cosines[2]
        # Snell's law
        else:
            self.directional_cosines[2] = np.sign(self.directional_cosines[2]) * np.sqrt(1 - sin_theta_t ** 2)
        self.directional_cosines /= np.linalg.norm(self.directional_cosines)

    # For simplicity, the reflected fraction will not be tracked any further for now (to avoid recursive photons). It
    # will just be simplified to continue in the exact same direction to infinity
    def reflect(self, interface):
        specular_reflection = abs(
            (interface[1].n - interface[0].n) /
            (interface[1].n + interface[0].n)
        ) ** 2
        # If the reflected fraction will be reflected out, add it to reflected count,
        if self.directional_cosines[2] > 0:
            self.R += self.weight * specular_reflection
        # Else add it to transmitted
        else:
            self.T += self.weight * specular_reflection
        self.weight = self.weight * (1 - specular_reflection)

    def absorb(self):
        self.A += self.weight * self.medium.albedo
        self.weight = self.weight - (self.weight * self.medium.albedo)

    def scatter(self):
        # Sample random scattering angles from distribution
        [xi, zeta] = np.random.rand(2)
        if self.medium.g != 0.0:
            lead_coeff = 1 / (2 * self.medium.g)
            term_2 = self.medium.g ** 2
            term_3 = (1 - term_2) / (1 - self.medium.g + (2 * self.medium.g * xi))
            cosine_theta = lead_coeff * (1 + term_2 - term_3)
        else:
            cosine_theta = (2 * xi) - 1
        theta = np.arccos(cosine_theta)
        phi = 2 * np.pi * zeta

        # Update direction cosines
        mu_x, mu_y, mu_z = self.directional_cosines

        if np.abs(mu_z) > 0.999:
            self.directional_cosines[0] = np.sin(theta) * np.cos(phi)
            self.directional_cosines[1] = np.sin(theta) * np.sin(phi)
            self.directional_cosines[2] = np.sign(mu_z) * np.cos(theta)
        else:
            deno = np.sqrt(1 - (mu_z ** 2))
            numr1 = (mu_x * mu_y * np.cos(phi)) - (mu_y * np.sin(phi))
            self.directional_cosines[0] = (np.sin(theta) * (numr1 / deno)) + (mu_x * np.cos(theta))
            numr2 = (mu_y * mu_z * np.cos(phi)) + (mu_x * np.sin(phi))
            self.directional_cosines[1] = (np.sin(theta) * (numr2 / deno)) + (mu_y * np.cos(theta))
            self.directional_cosines[2] = -(np.sin(theta) * np.cos(phi) * deno) + mu_z * np.cos(theta)

        # Explicit normalization to solve varied precision
        self.directional_cosines /= np.linalg.norm(self.directional_cosines)

    def plot_path(self, project_onto=None, axes=None, ignore_outside=True):
        project_onto = ['xz', 'yz', 'xy'] if project_onto == 'all' else project_onto
        project_onto = [project_onto] if not isinstance(project_onto, (list, tuple)) else project_onto
        data = {'x': [], 'y': [], 'z': []}
        for loc in self.location_history:
            if ignore_outside and (loc[2] < self.system.boundaries[0] or loc[2] > self.system.boundaries[-1]):
                break
            data['x'].append(loc[0])
            data['y'].append(loc[1])
            data['z'].append(loc[2])

        (_, axes) = plt.subplots(1, len(project_onto), figsize=(24, 6)) if axes is None else ([], axes)
        for ax, projection in zip(axes, project_onto):
            x = data[projection[0]]
            y = data[projection[1]]
            ax.plot(x, y, label=projection)
            ax.set_title(f'Projected onto {projection}-plane')
            ax.set_xlabel(f'Photon Displacement in {projection[0]}-direction (cm)')
            ax.set_ylabel(f'Photon Displacement in {projection[1]}-direction (cm)')

        return (plt.gcf(), axes)


class OpticalMedium:

    ## TODO: add metod for mu_s and mu_a to return coefficients as funcitons of photon wavelength
    def __init__(self, n=1, mu_s=1, mu_a=0, g=1, type='default', display_color=None):
        self.type = type
        self.n = n
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.g = g
        self.color = display_color

    @property
    def mu_t(self):
        return self.mu_s + self.mu_a

    @property
    def albedo(self):
        return self.mu_a / self.mu_t


# TODO: write function to take data from the MC simulation and insert into a database to be used at fitting. This should
#  include overwrite options for when simulation parameters match what has already been inserted, in the case of
#  updates.
def insert_into_mclut_database(simulation_parameters, simulation_results, db_file='mclut.db'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sO2 REAL,
        tHb REAL,
        wavelength REAL,
        g REAL,
        transmission REAL,
        reflectance REAL,
        absorption REAL
    )
    """)

    # Insert the simulation data
    cursor.execute("""
    INSERT INTO simulation_results (sO2, tHb, wavelength, g, transmission, reflectance, absorption)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (*simulation_parameters, *simulation_results))

    conn.commit()
    conn.close()