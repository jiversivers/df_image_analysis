import copy
import sqlite3
import warnings

from matplotlib import pyplot as plt, animation

from my_modules.monte_carlo.hardware import ring_pattern, cone_of_acceptance, ID, OD, THETA
from my_modules.image_processing.models import calculate_mus

try:
    import cupy as np
    from cupy import iterable

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError):
    import numpy as np
    from numpy import iterable

    np.is_available = lambda: False  # Mock the `is_available` method for consistency

# Get some default properties
mu_s, mu_a, wl = calculate_mus()


class System:
    def __init__(self, *args, surrounding_n=1, illuminator=None, detector=(None, 0)):
        """
        Create a system of optical mediums and its surroundings that hold the optical properties and can determine the
        medium of a location as well as interface crossing given two locations. The blocks are constructed top down in
        the order they are received from the top down (positive z is downward) starting at 0 and surrounded by infinite
        surroundings.

        ### Process
        1. Create the surroundings using the input or default n
        2. Stack the surroundings from z=negative infinity to z=0
        3. Iterate through all *args, and create the following:
            - Dict of the system stack with OpticalMedium object keys and len=2 list of boundaries for respective
              object including surroundings
            - List of OpticalMedium object layers in order of stacking including surroundings
            - Ndarray of the boundary (i.e. interface) z location between OpticalMedium objects, excluding +- infinite
        4. Add surroundings to the bottom from z=system thickness to z= positive infinity if the last layer was not
        semi-infinite

        ### Paramaters
        :param *args: A variable number of ordered pairs of OpticalMedium objects and their respective thickness (in
        that order). The number of args input must be even.
        """
        self.illuminator = illuminator

        self.detector = detector[0]
        self.detector_location = detector[1]

        assert len(args) % 2 == 0, "Arguments must be in groups of 2: medium object and thickness."
        self.surroundings = OpticalMedium(n=surrounding_n, mu_s=0, mu_a=0, name='surroundings')

        # Add surroundings from -inf to 0
        interface = 0  # Current interface location during stacking
        self.stack = {(float('-inf'), interface): self.surroundings}  # Dict with tuple layer boundaries: layer object
        self.layer = [self.surroundings]  # List of layers in order of addition
        boundaries = []  # List of boundary depths between layers

        # Iterate through args to stack layers
        for i in range(0, len(args), 2):
            d = args[i + 1]
            d = float(d)
            self.stack[(interface, interface + d)] = args[i]
            self.layer.append(args[i])
            boundaries.append(interface)
            interface += d

        # Last layer finish
        boundaries.append(interface)
        self.boundaries = np.asarray(boundaries)

        # Add surroundings if not semi-infinte
        if interface < float('inf'):
            boundaries.append(interface)
            self.stack[(interface, float('inf'))] = self.surroundings

        self.boundaries = np.asarray(boundaries)

    def __repr__(self):
        return ''.join([f' <-- {bound[0]} --> {layer}' for bound, layer in self.stack.items()])

    def __str__(self):
        stack = '\n'
        space = 0
        for bound, layer in self.stack.items():
            txt = layer.__str__()
            space = len(txt) if len(txt) > space else space
            txt = f'{bound[0]:.4g}'
            space = len(txt) if len(txt) > space else space
        txt = f'{bound[1]:.4g}'
        space = len(txt) if len(txt) > space else space
        space += 8

        for bound, layer in self.stack.items():
            txt = f' {bound[0]:.4g} '
            lfill = '-' * ((space - len(txt) - 2) // 2)
            rfill = '-' * (space - len(txt) - len(lfill) - 2)
            boundary = f'|{lfill}{txt}{rfill}|\n'

            txt = f' {layer} '
            lfill = ' ' * int(np.floor((space - len(txt) - 6) / 2))
            rfill = ' ' * (space - len(txt) - len(lfill) - 6)
            layer = f'|->{lfill}{txt}{rfill}<-|\n'
            stack += boundary + layer

        txt = f' {bound[1]:.4g} '
        lfill = '-' * ((space - len(txt) - 2) // 2)
        rfill = '-' * (space - len(txt) - len(lfill) - 2)
        boundary = f'|{lfill}{txt}{rfill}|\n'
        stack += boundary
        border = ' ' + '_' * (space - 2) + ' '
        return border + stack + border

    def beam(self, **kwargs):
        photon = self.illuminator.photon()
        photon.system = self
        for key, val in kwargs.items():
            setattr(photon, key, val)
        return photon

    def in_medium(self, location):
        """
        Return the medium(s) that are at the queried coordinate. If the coordinate is an interface location, the mediums
        that makeup the interface are returned as a tuple, this includes boundary interfaces being returned with False
        on the surroundings side.

        ### Process
        1. Get the z-coordinate from the input
        2. Check it against boundaries of each medium of the system until it is found that either:
            - It is between any of the boundaries, it is in the medium within those boundaries
            - It is at a boundary, it is "in" the two mediums that make up that interface
        3. Break and return the medium(s) of the queried point.

        ### Parameters:
        :param location: (tuple, list, ndarray, float, or int) The coordinates or z-coordinate to query.

        ### Returns
        :return in_: (medium or tuple of mediums): The medium of the queried z-coordinate or a tuple of interfaces if
                     the coordinate is at an interface
        """

        z = location[2] if iterable(location) else location
        if z == float('-inf'):
            return self.layer[0]
        elif z == float('inf'):
            return self.layer[-1]

        for bound, medium in self.stack.items():
            if bound[0] < z < bound[1]:
                in_ = medium
                break
            elif z == bound[0]:
                # Determine the mediums on either side by simply bumping the coordinates forward and backward slightly
                in_ = (
                    self.in_medium(np.nextafter(z, float('-inf'))),
                    self.in_medium(np.nextafter(z, float('inf')))
                )
                break
        return in_

    def interface_crossed(self, location1, location2):
        """
        Determines the first interface crossed when moving between two locations, considering only the z-coordinates.

        This method checks if any interface boundaries lie between the given z-coordinates. If an interface is crossed,
        the method calculates its z-location and identifies the two mediums forming the interface.

        ### Process:
        1. Identify boundaries that fall between the start and end z-coordinates.
        2. Compute the distance from the start z-coordinate to each boundary.
        3. Apply a mask to filter boundaries that are actually crossed.
        4. Select the closest crossed boundary as the interface plane.
        5. Determine the two mediums making up the interface by slightly shifting the plane's z-coordinate:
           - Backwards (toward the start) to find the first medium.
           - Forwards (away from the start) to find the second medium.

        ### Parameters:
        - :param location1: Starting location of the query.
        - :param location2: Ending location of the query.

        ### Returns:
        - :return interface: (tuple): The two media forming the crossed interface, or an empty list `[]` if no
                              interface is crossed.
        - :return plane: (float or bool): The z-coordinate of the crossed interface if one is found, otherwise `False`.
        """

        # Get z coords
        z = []
        for location in [location1, location2]:
            z.append(location[2] if iterable(location) else location)

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

        # Check if any boundaries fall between the zs and put into nd boolean array to use as a mask
        crossed = np.array([z_sorted[0] < bound < z_sorted[1] for bound in self.boundaries], dtype=bool)

        # Determine the distance of each boundary from the start coordinate
        dist_from_start = np.abs(self.boundaries - z[0], dtype=float)

        dist_from_start[~crossed] = float('inf')  # Set uncrossed distance to inf so they are not considered
        if np.any(crossed):
            # Set the closest crossed boundary to the plane
            plane = self.boundaries[np.argmin(dist_from_start)]
        else:
            plane = None

        # If an interface is crossed, get the two mediums that make it up
        interface = []
        if plane is not None:
            # Determine the mediums on either side by simply bumping the coordinates forward and backward slightly
            interface.append(self.in_medium(np.nextafter(plane, z[0])))
            interface.append(self.in_medium(np.nextafter(plane, z[1])))

        return tuple(interface), plane

    def represent_on_axis(self, ax=None):
        if ax is None:
            ax = plt.gca()
        lim = [
            -0.1 * self.boundaries[-1], 1.1 * self.boundaries[-1]
        ] if self.boundaries[-1] != float('inf') else [
            -0.1 * ax.get_ylim()[0], 1.1 * ax.get_ylim()[1]
        ]
        if ax.name == '3d':
            ax.set(zlim=lim)
        else:
            # ax.set(ylim=lim)
            alpha = 0.0
            for bound, medium in self.stack.items():
                depth = np.diff(bound)
                y_edge = bound[0] - 0.1 * depth
                x_edge = ax.set_xlim(ax.get_xlim())[0] * 0.95
                ax.text(x_edge, y_edge, medium.name, fontsize=12)
                line_x = 100 * np.asarray(ax.get_xlim())
                alpha += 0.2
                ax.fill_between(line_x, bound[0], bound[1],
                                color='gray' if medium.display_color is None else medium.display_color,
                                alpha=alpha if medium.display_color is None else 1)


class IndexableProperty(np.ndarray):
    def __new__(cls, arr, normalize=False):
        obj = np.asarray(arr, dtype=np.float64).view(cls)
        obj.normalize = normalize
        obj /= np.linalg.norm(obj) if normalize else 1
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self._normalize = getattr(obj, '_normalize', False)

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        self._normalize = value
        if self._normalize:
            self /= np.linalg.norm(self)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if self.normalize:
            self /= np.linalg.norm(self)


class Photon:

    def __init__(self, wavelength,
                 system=None,
                 directional_cosines=(0, 0, 1),
                 location_coordinates=(0, 0, 0),
                 weight=1,
                 russian_roulette_constant=20,
                 recurse=True,
                 recursion_depth=0,
                 recursion_limit=100,
                 throw_recursion_error=True,
                 keep_secondary_photons=False,
                 tir_limit=float('inf')):

        # Init current_photon state
        self.wavelength = wavelength
        self.system = system
        self.directional_cosines = directional_cosines
        self.location_coordinates = location_coordinates
        self.exit_location = None
        self.exit_direction = None
        self.exit_weight = None
        self._weight = float(weight)
        self.russian_roulette_constant = russian_roulette_constant
        self._medium = None
        self.recurse = recurse
        self.recursion_depth = recursion_depth
        self.recursion_limit = recursion_limit
        self.throw_recursion_error = throw_recursion_error
        self.keep_secondary_photons = keep_secondary_photons
        self.secondary_photons = []
        self.tir_limit = tir_limit

        # Call setter in case current_photon is DOA
        self.weight = weight

        # Init trackers
        self.location_history = [(self.location_coordinates, self.weight)]
        self.recursed_photons = 0
        self.tir_count = 0
        self.A = 0.0
        self.T = 0.0
        self.R = 0.0

    # EXPLANATION OF INDEXABLE_PROPERTIES
    # When one of these properties, obj, is used in the form obj = value, it will call the setter, which sets the
    # attribute to an indexable_property object with value. When it is used in the form obj[i] = val, the getter is
    # called to return obj. Then, the indexable_property __setitem__ is called to update obj. This should ensure that
    # these properties are always np.ndarrays and (when set) are normalized.
    @property
    def directional_cosines(self):
        return self._directional_cosines

    @directional_cosines.setter
    def directional_cosines(self, value):
        self._directional_cosines = IndexableProperty(value, normalize=True)

    @property
    def location_coordinates(self):
        return self._location_coordinates

    @location_coordinates.setter
    def location_coordinates(self, value):
        self._location_coordinates = IndexableProperty(value)

    def copy(self, wavelength=None, **kwargs):
        new_obj = copy.deepcopy(self)

        # Check for kwarg overwrites
        for key, value in kwargs.items():
            if hasattr(new_obj, key):
                setattr(new_obj, key, value)
            elif hasattr(new_obj, f'_{key}'):
                setattr(new_obj, f'_{key}', value)

        # Reset tracker attributes
        for key in ['T', 'R', 'A', 'tir_count', 'recursed_photons']:
            setattr(new_obj, key, 0)
        new_obj.location_history = [(new_obj.location_coordinates, new_obj.weight)]

        return new_obj

    def __repr__(self):
        out = ''
        for key, val in self.__dict__.items():
            out += f"{key.strip('_')}: {val}\n"
        return out

    def simulate(self):
        assert self.system is not None, RuntimeError('Photon must be in an Optical System object to simulate.')
        if self.recursion_depth >= self.recursion_limit:
            if not self.throw_recursion_error:
                self.recurse = False
                warnings.warn(
                    'Maximum recursion depth reached. Simulating deep photon with no recursion.\n'
                    'To throw and error at maximum depth instead, set throw_recursion_error to TRUE. This will throw an'
                    ' RecursionError instead of simulating without recursion.')
            else:
                RecursionError(
                    'Maximum photon recursion limit reached. Recursion depth limit can be increased with the '
                    'recursion_limit attribute.\n'
                    'To switch this error off and throw a warning instead, set throw_recursion_error to FALSE. This '
                    'will simulate the photon at the limit without recursion, rather than throwing an error.')
        while not self.is_terminated:
            self.absorb()
            self.move()
            self.scatter()

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight
        if 0 < weight < 0.005:
            self.russian_roulette()

    def russian_roulette(self):
        if np.random.rand() >= (1 / self.russian_roulette_constant):
            self._weight = 0
        else:
            self.weight = self.weight * self.russian_roulette_constant

    @property
    def medium(self):
        self._medium = self.system.in_medium(self.location_coordinates)
        if isinstance(self._medium, (list, tuple)):
            self._medium = self.headed_into
        return self._medium

    @property
    def is_terminated(self):
        self._is_terminated = (self.medium == self.system.surroundings or self.weight <= 0.0)
        return self._is_terminated

    @is_terminated.setter
    def is_terminated(self, value):
        self._is_terminated = value

    @property
    def headed_into(self):
        bumped_coords = self.location_coordinates
        medium = []
        while isinstance(medium, (list, tuple)):
            medium = self.system.in_medium(bumped_coords)
            # Increment coordinates smallest possible amount in same direction of the current_photon
            bumped_coords = np.nextafter(bumped_coords, bumped_coords + self.directional_cosines)
        return medium

    def move(self, step=None):
        while True:
            # Get current state
            mu_t = self.medium.mu_t_at(self.wavelength)
            dir_cos = self.directional_cosines
            loc = self.location_coordinates

            # Compute propagation step
            if mu_t > 0:
                # If scattering occurs, step is sampled from the distribution
                step = -np.log(np.random.random()) / mu_t if step is None else step
            else:
                # If in a medium with mu_t = 0, force move to next interface
                step = float('inf')

            # If interface will be crossed, go to it, reflect/refract and restart
            interface, plane = self.system.interface_crossed(loc, loc + step * dir_cos)
            if interface and plane is not None:
                # Compute step to interface
                interface_step = (plane - loc[2]) / dir_cos[2] if dir_cos[2] != 0 else float('inf')

                if interface_step < step:
                    # Move to the interface
                    self.location_coordinates = loc + interface_step * dir_cos

                    # Handle reflection/refraction
                    self.reflect_refract(interface)

                    # Update history
                    self.location_history.append((self.location_coordinates, self.weight))

                    # Restart loop
                    step = None
                    continue

            # If no interface is crossed, update location
            self.location_coordinates = loc + step * dir_cos

            # Check if this got just to an interface and need reflection/refraction
            interface = self.system.in_medium(self.location_coordinates)
            if isinstance(interface, tuple):
                self.reflect_refract(interface)

            # Update history and end loop
            self.location_history.append((self.location_coordinates, self.weight))
            break

        # Check termination (exiting the system)
        if not (self.system.boundaries[0] < self.location_coordinates[2] < self.system.boundaries[-1]):
            self.exit_location = self.location_history[-2][0] if len(
                self.location_history) > 1 else self.location_coordinates
            self.exit_weight = self.weight

            # Detector check
            if self.system.detector and self.exit_location[2] == self.system.detector_location:
                self.system.detector(self)

            # Reflection or transmission
            if self.location_coordinates[2] < self.system.boundaries[0]:  # Reflection
                self.R += self.weight
            elif self.location_coordinates[2] > self.system.boundaries[-1]:  # Transmission
                self.T += self.weight

            # Photon is now terminated
            self.weight = 0

    def reflect_refract(self, interface):
        # Get incidence state
        mu_x, mu_y, mu_z_i = self.directional_cosines
        n1 = interface[0].n
        n2 = interface[1].n

        # Calculate refraction
        sin_theta_t = n1 / n2 * np.sqrt(1 - mu_z_i ** 2)

        # TIR
        if sin_theta_t > 1:
            mu_z_t = -mu_z_i
            self.tir_count += 1
            if self.tir_count > self.tir_limit:
                self.A += self.weight
                self.weight = 0

        # Snell's law + Fresnel
        else:
            mu_z_t = np.sqrt(1 - sin_theta_t ** 2)

            # Calculate reflection
            rs = np.abs(((n1 * np.abs(mu_z_i)) - (n2 * mu_z_t)) / ((n1 * np.abs(mu_z_i)) + (n2 * mu_z_t))) ** 2
            rp = np.abs(((n2 * mu_z_t) - (n1 * np.abs(mu_z_i))) / ((n1 * np.abs(mu_z_i)) + (n2 * mu_z_t))) ** 2
            specular_reflection = 0.5 * (rs + rp) * self.weight

            mu_z_t = np.sign(mu_z_i) * mu_z_t  # Ensure correct sign

            # Inject secondary current_photon to account for reflected portion with current direciton and location
            if self.recurse and self.recursion_depth <= self.recursion_limit:
                secondary_photon = self.copy(directional_cosines=(mu_x, mu_y, -mu_z_i),
                                             weight=specular_reflection,
                                             recursion_depth=self.recursion_depth + 1)
                recursion_error = None
                try:
                    secondary_photon.simulate()
                except RecursionError as e:
                    recursion_error = e
                finally:
                    if self.keep_secondary_photons:
                        self.secondary_photons.append(secondary_photon)
                        for photon in secondary_photon.secondary_photons:
                            self.secondary_photons.append(photon)
                    self.recursed_photons = secondary_photon.recursed_photons + 1
                    if recursion_error is not None:
                        raise recursion_error

                self.T += secondary_photon.T
                self.R += secondary_photon.R
                self.A += secondary_photon.A

            else:
                # If the reflected fraction will be reflected out, add it to reflected count,
                if mu_z_i > 0:
                    self.R += specular_reflection
                # Else add it to transmitted
                elif mu_z_i < 0:
                    self.T += specular_reflection

            # Updated for transmitted portion
            mu_x *= n1 / n2
            mu_y *= n1 / n2
            self.weight = self.weight - specular_reflection

        # Send to setter for normalization
        self.directional_cosines = [mu_x, mu_y, mu_z_t]

    # TODO: Add support for fluorescence-based secondary photons
    def absorb(self):
        absorbed_weight = self.weight * self.medium.albedo_at(self.wavelength)
        self.A += absorbed_weight
        self.weight = self.weight - absorbed_weight

    def scatter(self, theta_phi=None):
        if theta_phi is None:
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
        else:
            theta, phi = theta_phi

        # Update direction cosines
        mu_x, mu_y, mu_z = self.directional_cosines
        new_directional_cosines = np.array([0, 0, 0], dtype=np.float64)
        if np.abs(mu_z) > 0.999:
            new_directional_cosines[0] = np.sin(theta) * np.cos(phi)
            new_directional_cosines[1] = np.sin(theta) * np.sin(phi)
            new_directional_cosines[2] = np.sign(mu_z) * np.cos(theta)
        else:
            deno = np.sqrt(1 - (mu_z ** 2))
            numr1 = (mu_x * mu_y * np.cos(phi)) - (mu_y * np.sin(phi))
            new_directional_cosines[0] = (np.sin(theta) * (numr1 / deno)) + (mu_x * np.cos(theta))
            numr2 = (mu_y * mu_z * np.cos(phi)) + (mu_x * np.sin(phi))
            new_directional_cosines[1] = (np.sin(theta) * (numr2 / deno)) + (mu_y * np.cos(theta))
            new_directional_cosines[2] = -(np.sin(theta) * np.cos(phi) * deno) + mu_z * np.cos(theta)

        # Update directional cosines with new direciton (done at once for normalization consistency)
        self.directional_cosines = new_directional_cosines

    def plot_path(self, project_onto=None, axes=None, ignore_outside=True):
        project_onto = ['xz', 'yz', 'xy'] if project_onto == 'all' else project_onto
        project_onto = [project_onto] if isinstance(project_onto, (str)) or project_onto is None else project_onto
        data = {'x': [], 'y': [], 'z': []}
        for loc in self.location_history:
            if ignore_outside and (loc[0][2] < self.system.boundaries[0] or loc[0][2] > self.system.boundaries[-1]):
                break
            data['x'].append(loc[0][0])
            data['y'].append(loc[0][1])
            data['z'].append(loc[0][2])

        fig = plt.figure(figsize=(8 * len(project_onto), 8)) if not plt.get_fignums() else plt.gcf()
        if project_onto[0]:
            axes = [fig.add_subplot(1, len(project_onto), i + 1) for i in
                    range(len(project_onto))] if axes is None else axes
            for ax, projection in zip(axes, project_onto):
                x = data[projection[0]]
                y = data[projection[1]]
                ax.plot(x, y, label=projection)
                ax.set_title(f'Projected onto {projection}-plane')
                ax.set_xlabel(f'Photon Displacement in {projection[0]}-direction (cm)')
                ax.set_ylabel(f'Photon Displacement in {projection[1]}-direction (cm)')
                if projection[1] == 'z' and not ax.yaxis_inverted():
                    ax.invert_yaxis()
        else:
            axes = fig.add_subplot(projection='3d') if axes is None else axes
            axes.plot(data['x'], data['y'], data['z'])
            axes.set_title(f'Photon Path')
            axes.set(xlabel=f'Photon Displacement in x-direction (cm)')
            axes.set(ylabel=f'Photon Displacement in y-direction (cm)')
            axes.set(zlabel=f'Photon Displacement in z-direction (cm)')
            if not axes.zaxis_inverted():
                axes.invert_zaxis()

        return fig, axes

    def animate_path(self, ax=None, filename=None):
        filename = 'animation.gif' if filename is None else filename

        def update_lines(num, paths, lines):
            for line, path in zip(lines, paths):
                line.set_data_3d(path[:num])
            return lines

        fig = plt.figure() if not plt.get_fignums() else plt.gcf()
        ax = fig.add_subplot(projection='3d') if ax is None else ax
        lines = [ax.plot([], [], [])[0] for _ in self.location_history]
        ani = animation.FuncAnimation(
            fig, update_lines, len(self.location_history), fargs=(self.location_history, lines), interval=100
        )
        ani.save(filename)


class OpticalMedium:

    def __init__(self, n=1, mu_s=mu_s, mu_a=mu_a, wavelengths=wl, g=1,
                 name='default', display_color=None):
        self.name = name
        self.n = n
        self.mu_s = np.array(mu_s)
        self.mu_a = np.array(mu_a)
        self.wavelengths = np.array(wavelengths)
        self.g = g
        self.display_color = display_color

    def __repr__(self):
        return self.name.capitalize() + ' Optical Medium Object'

    def __str__(self):
        if self.name == 'default':
            return f'Optical Medium: n={self.n}, mu_s={self.mu_s}, mu_a={self.mu_a}, g={self.g}'
        return self.name.capitalize()

    def _wave_index(self, wavelength):
        if iterable(wavelength) and iterable(self.wavelengths):
            return [np.where(self.wavelengths == wl)[0][0] for wl in wavelength]
        elif iterable(self.wavelengths):
            return np.where(self.wavelengths == wavelength)[0][0]
        else:
            return 0

    def mu_s_at(self, wavelengths):
        if iterable(self.mu_s):
            return np.interp(wavelengths, self.wavelengths, self.mu_s)
        return mu_s

    def mu_a_at(self, wavelengths):
        if iterable(self.mu_a):
            return np.interp(wavelengths, self.wavelengths, self.mu_a)
        return self.mu_a

    def mu_t_at(self, wavelengths):
        if iterable(self.mu_t):
            return np.interp(wavelengths, self.wavelengths, self.mu_t)
        return self.mu_t

    def albedo_at(self, wavelengths):
        if iterable(self.albedo):
            return np.interp(wavelengths, self.wavelengths, self.albedo)
        return self.albedo

    @property
    def mu_t(self):
        return self.mu_s + self.mu_a

    @property
    def albedo(self):
        if self.mu_t == 0:
            return 0
        else:
            return self.mu_a / self.mu_t


def insert_into_mclut_database(simulation_parameters, simulation_results, db_file='mclut.db'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mu_s REAL,
        mu_a REAL,
        g REAL,
        depth REAL,
        transmission REAL,
        reflectance REAL,
        absorption REAL,
        UNIQUE (mu_s, mu_a, g, depth)
    )
    """)

    # Insert the simulation data
    cursor.execute("""
    INSERT INTO simulation_results (mu_s, mu_a, g, depth, transmission, reflectance, absorption)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (*simulation_parameters, *simulation_results))

    conn.commit()
    conn.close()


def sample_spectrum(wavelengths, spectrum):
    # Normalize PDF
    spectrum /= np.sum(spectrum)

    # Compute CDF
    cdf = np.cumsum(spectrum)

    # Take random sample
    i = np.random.uniform(0, 1)

    # Interpolate value of sample from CDF
    return np.interp(i, cdf, wavelengths)


class Illumination:
    def __init__(self,
                 pattern=ring_pattern((ID, OD), THETA),
                 spectrum=None):
        self.pattern = pattern
        self.spectrum = spectrum

    def photon(self):
        location, direciton = self.pattern()
        wavelength = sample_spectrum(self.spectrum) if self.spectrum else None
        return Photon(wavelength, location_coordinates=location, directional_cosines=direciton)


class Detector:
    def __init__(self, acceptor=cone_of_acceptance(ID)):
        self.acceptor = acceptor
        self.n_total = 0
        self.n_detected = 0

    def detect(self, location, direction, weight=None):
        weight = weight if weight is not None else 1
        self.n_total += weight
        x, y = location[:2]
        mu_z = direction[-1] if iterable(direction) else direction
        if self.acceptor(x, y, mu_z=mu_z):
            self.n_detected += weight

    def __call__(self, photon):
        assert isinstance(photon, Photon), ValueError('Detector object can only be called directly with a photon. '
                                                      'Use detector.detect() for non-photon test cases.')
        self.detect(photon.exit_location, photon.exit_direction, photon.exit_weight)

    def reset(self):
        self.n_total = 0
        self.n_detected = 0
