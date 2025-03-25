from __future__ import annotations

import copy
import warnings
from numbers import Real
from typing import Union, Optional, Iterable, Tuple, Callable, List, Dict, Any

from matplotlib import pyplot as plt, animation

from .hardware import ring_pattern, cone_of_acceptance, ID, OD, THETA
from .utils import calculate_mus, sample_spectrum

try:
    import cupy as np
    from cupy import iterable
    from cupy.typing import NDArray

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError):
    import numpy as np
    from numpy import iterable
    from numpy.typing import NDArray

    np.is_available = lambda: False  # Mock the `is_available` method for consistency

# Get some default properties
mu_s, mu_a, wl = calculate_mus()


class Medium:
    def __init__(self,
                 n: Real = 1,
                 mu_s: Union[Real, Iterable[Real]] = 0,
                 mu_a: Union[Real, Iterable[Real]] = 0,
                 wavelengths: Iterable[Real] = wl,
                 ref_wavelength: Real = 650,
                 g: Real = 1,
                 desc: str = 'default',
                 display_color: Optional[str] = None):
        self.desc = desc
        self.n = n
        self.mu_s = np.array(mu_s)
        self.mu_a = np.array(mu_a)
        self.wavelengths = np.array(wavelengths)
        self.g = g
        self.display_color = display_color
        self.ref_wavelength = ref_wavelength

    def __repr__(self):
        return self.desc.capitalize() + ' Optical Medium Object'

    def __str__(self):
        if self.desc == 'default':
            return f'Optical Medium: n={self.n}, mu_s={self.mu_s}, mu_a={self.mu_a}, g={self.g}'
        return self.desc.capitalize()

    def set(self, **kwargs: Any) -> None:
        """
        Sets multiple attributes of the object in bulk.

        This method is useful for updating the medium's attributes dynamically, particularly when modifying properties
        during iterative simulations while generating a lookup table. It assigns the provided keyword arguments to
        existing attributes of the object.

        :param kwargs: A dictionary of attribute names and their corresponding new values.
        :type kwargs: dict
        :return: None
        :raises AttributeError: If a provided attribute name does not exist.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _wave_index(self, wavelength: Union[Real, Iterable[Real]]):
        if iterable(wavelength) and iterable(self.wavelengths):
            return [np.where(self.wavelengths == wl)[0][0] for wl in wavelength]
        elif iterable(self.wavelengths):
            return np.where(self.wavelengths == wavelength)[0][0]
        else:
            return 0

    def mu_s_at(self, wavelengths: Real):
        if iterable(self.mu_s):
            return np.interp(wavelengths, self.wavelengths, self.mu_s)
        return self.mu_s

    def mu_a_at(self, wavelengths: Real):
        if iterable(self.mu_a):
            return np.interp(wavelengths, self.wavelengths, self.mu_a)
        return self.mu_a

    def mu_t_at(self, wavelengths: Real):
        if iterable(self.mu_t):
            return np.interp(wavelengths, self.wavelengths, self.mu_t)
        return self.mu_t

    def albedo_at(self, wavelengths: Real):
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

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        if np.any(self.mu_s == 0 and g != 1):
            warnings.warn('g is automatically set to 1 where mu_s is 0. '
                          'Set a non-zero scattering coefficient if a non-unity g value is necessary.')
            self._g = np.where(self.mu_s == 0, 1, g)
        self._g = g


class Illumination:
    def __init__(self,
                 pattern: Callable = ring_pattern((ID, OD), THETA),
                 spectrum: Optional[Iterable] = None) -> None:
        self.pattern = pattern
        self.spectrum = spectrum

    def photon(self, batch_size: int = 50000, **kwargs: Any) -> Photon:
        location, direction = self.pattern(batch_size)
        wavelength = sample_spectrum(self.spectrum) if self.spectrum else None
        return Photon(wavelength, batch_size=batch_size, location_coordinates=location, directional_cosines=direction,
                      **kwargs)


class Detector:
    def __init__(self, acceptor: Callable = cone_of_acceptance(ID), desc: Optional[str] = 'default') -> None:
        self.acceptor = acceptor
        self.n_total = 0
        self.n_detected = 0
        self.desc = desc

    def detect(self, location: Iterable[Real],
               direction: Union[Real, Iterable[Real]],
               weights: Optional[Union[Real, Iterable[Real]]] = None) -> None:
        weights = weights if weights is not None else 1
        self.n_total += np.nansum(weights)
        x, y = location[:, :2].T
        mu_z = direction[:, -1]
        accepted_mask = self.acceptor(x, y, mu_z=mu_z)
        self.n_detected += np.nansum(weights[accepted_mask])

    def __call__(self,
                 photon: Photon,
                 mask: Optional[NDArray[np.bool_]] = True) -> None:
        if not isinstance(photon, Photon):
            raise ValueError('Detector object can only be called directly with a photon. '
                             'Use detector.detect() for non-photon test cases.')
        self.detect(photon.exit_location[mask], photon.exit_direction[mask], photon.exit_weights[mask])

    def reset(self) -> None:
        self.n_total = 0
        self.n_detected = 0


# Default "sampler" to start photons straight down at origin
pencil_beam = Illumination(lambda n: (np.repeat((0, 0, 0), n), np.repeat((0, 0, 1), n)))


class System:
    def __init__(self, *args,
                 surrounding_n: Real = 1,
                 illuminator: Optional[Illumination] = pencil_beam,
                 detector: Optional[Tuple[Detector, Real]] = (None, None)) -> None:
        """
        Create a system of optical mediums and its surroundings that hold the optical properties and can determine the
        medium of a location as well as interfaces crossing given two locations. The blocks are constructed top down in
        the order they are received from the top down (positive z is downward) starting at 0 and surrounded by infinite
        surroundings.

        ### Process
        1. Create the surroundings using the input or default n
        2. Stack the surroundings from z=negative infinity to z=0
        3. Iterate through all *args, and create the following:
            - Dict of the system stack with Medium object keys and len=2 list of boundaries for respective
              object including surroundings
            - List of Medium object layers in order of stacking including surroundings
            - Ndarray of the boundary (i.e. interfaces) z location between Medium objects, excluding +- infinite
        4. Add surroundings to the bottom from z=system thickness to z= positive infinity if the last layer was not
        semi-infinite

        ### Paramaters
        :param *args: A variable number of ordered pairs of Medium objects and their respective thickness (in
        that order). The number of args input must be even.
        """
        self.illuminator = illuminator

        self.detector = detector[0]
        self.detector_location = detector[1]
        self._boundaries = [float('-inf'), 0, float('inf')]

        if not len(args) % 2 == 0:
            raise ValueError("Arguments must be in groups of 2: medium object and thickness.")

        # Add surroundings from -inf to 0 to inf
        self.surroundings = Medium(n=surrounding_n, mu_s=0, mu_a=0, desc='surroundings')
        self.layer = [self.surroundings, self.surroundings]  # List of layers in order of addition

        # Iterate through args to stack layers
        for i in range(0, len(args), 2):
            self.add(args[i], args[i + 1])

    @property
    def boundaries(self) -> NDArray[Real]:
        return np.asarray(self._boundaries)

    @boundaries.setter
    def boundaries(self, value: List):
        self._boundaries = value

    def __repr__(self) -> str:
        return '|'.join(
            [f' <-- {bound[0]} --> {layer}' for bound, layer in self.stack.items()]) + f' <-- {self.boundaries[-1]}|'

    def __str__(self) -> str:
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

    @property
    def stack(self) -> Dict[Tuple[float, float], Medium]:
        """
        Provides an up-to-date dictionary representation of layer boundaries.

        This property dynamically constructs a dictionary mapping layer boundaries to their corresponding
        layers. It is useful for querying interfaces efficiently in real-time.

        :return: A dictionary where keys are tuples representing layer boundaries (lower_bound, upper_bound),
                 and values are the corresponding layers.
        :rtype: dict[tuple[Real, Real], Layer]
        """
        return {(self.boundaries[i], self.boundaries[i + 1]): layer for i, layer in enumerate(self.layer)}

    def add(self, layer: Medium, depth: Real) -> None:
        """
        Adds a new layer of a given depth to the system.

        This method inserts a new `Medium` layer at the specified depth. If the preceding layer was
        semi-infinite, an error is raised. The method automatically replaces the surrounding filler layer
        before adding the new layer and ensures that the system remains properly bounded, extending back
        to infinity if necessary.

        :param layer: The `Medium` object representing the new layer.
        :type layer: Medium
        :param depth: The thickness of the new layer.
        :type depth: Real
        :raises ValueError: If the preceding layer is semi-infinite.
        :return: None
        """

        if not (self.layer[-1] == self.surroundings or self.boundaries[-1] != float('inf')):
            raise OverflowError('Cannot add to semi-infinite system.')
        depth = float(depth)

        # Get the bound of the last layer before surroundings
        replacement_idx = -2 if self.boundaries[-1] == float('inf') else -1
        start_interface = self.boundaries[replacement_idx]
        old_boundaries = self.boundaries[:replacement_idx]
        end_interface = start_interface + depth
        if replacement_idx == -2:
            self.layer[-1] = layer
        else:
            self.layer.append(layer)
        self.boundaries = np.append(old_boundaries, [start_interface, end_interface])

        # Add surroundings if not semi-infinte
        if end_interface < float('inf'):
            self.add(self.surroundings, float('inf'))

    def beam(self, batch_size: int = 50000, **kwargs: Any) -> Photon:
        photon = self.illuminator.photon(batch_size=batch_size, system=self)
        for key, val in kwargs.items():
            setattr(photon, key, val)
        return photon

    def in_medium(self, location: Iterable[Real]) -> NDArray[Union[Medium, Tuple[Medium, Medium]]]:
        """
        Return the medium(s) that are at the queried coordinate. If the coordinate is an interfaces location, the
        mediums that makeup the interfaces are returned as a tuple, this includes boundary interfaces being returned
        with False on the surroundings side.

        ### Process
        1. Get the z-coordinate from the input
        2. Check it against boundaries of each medium of the system until it is found that either:
            - It is between any of the boundaries, it is in the medium within those boundaries
            - It is at a boundary, it is "in" the two mediums that make up that interfaces
        3. Break and return the medium(s) of the queried point.

        ### Parameters:
        :param location: (tuple, list, ndarray, float, or int) The coordinates or z-coordinate to query.

        ### Returns
        :return in_: (medium or tuple of mediums): The medium of the queried z-coordinate or a tuple of interfaces if
                     the coordinate is at an interfaces
        """
        location = np.asarray(location)
        z = location if np.ndim(location) == 1 else np.asarray(location)[:, 2]
        in_medium = np.empty_like(z, dtype=object)
        in_medium = np.where(z == float('inf'), self.surroundings, in_medium)
        in_medium = np.where(z == float('-inf'), self.surroundings, in_medium)

        for bound, medium in self.stack.items():
            # IF between boundaries, get that medium
            mask_inside = np.logical_and(bound[0] < z, z < bound[1])
            mask_boundary = (bound[0] == z)

            in_medium[mask_inside] = medium

            # If at a boundary, get the mediums on each side
            if np.any(mask_boundary):
                z_neg_move = np.nextafter(z[mask_boundary], float('-inf'))
                z_pos_move = np.nextafter(z[mask_boundary], float('inf'))
                output1 = self.in_medium(z_neg_move)
                output2 = self.in_medium(z_pos_move)
                for i, idx in enumerate(np.where(mask_boundary)[0]):
                    in_medium[idx] = (output1[i], output2[i])

            # If all have been filled in, break
            if not np.any(np.equal(in_medium, None)):
                break
        return in_medium

    def interface_crossed(self,
                          location0: Iterable[Real],
                          location1: Iterable[Real]) -> Union[Tuple[Union[Medium, ()], Union[Real, None]]]:
        """
        Determines the first interfaces crossed when moving between two locations, considering only the z-coordinates.

        This method checks if any interfaces boundaries lie between the given z-coordinates. If an interfaces is crossed,
        the method calculates its z-location and identifies the two mediums forming the interfaces.

        ### Process:
        1. Identify boundaries that fall between the start and end z-coordinates.
        2. Compute the distance from the start z-coordinate to each boundary.
        3. Apply a mask to filter boundaries that are actually crossed.
        4. Select the closest crossed boundary as the interfaces plane.
        5. Determine the two mediums making up the interfaces by slightly shifting the plane's z-coordinate:
           - Backwards (toward the start) to find the first medium.
           - Forwards (away from the start) to find the second medium.

        ### Parameters:
        - :param location1: Starting location of the query.
        - :param location2: Ending location of the query.

        ### Returns:
        - :return interfaces: (tuple): The two media forming the crossed interfaces, or an empty list `[]` if no
                              interfaces is crossed.
        - :return plane: (float or bool): The z-coordinate of the crossed interfaces if one is found, otherwise `False`.
        """

        # Get z coords
        z0 = np.asarray(location0) if np.shape(location0)[0] == 1 else np.asarray(location0)[:, 2]
        z1 = np.asarray(location1) if np.shape(location1)[0] == 1 else np.asarray(location1)[:, 2]

        # Move off of boundaries (uncrossed!)
        is_boundary1 = np.isin(z0, self.boundaries)
        is_boundary2 = np.isin(z1, self.boundaries)

        z0 = np.where(is_boundary1, np.nextafter(z0, z1), z0)
        z1 = np.where(is_boundary2, np.nextafter(z1, z0), z1)

        # Sort for easier logic
        z_sorted = np.sort(np.stack((z0, z1), axis=-1), axis=-1)

        # Check if any boundaries fall between the zs and put into nd boolean array to use as a mask
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        crossed_mask = (z_sorted[..., 0, np.newaxis] < boundaries) & (boundaries < z_sorted[..., 1, np.newaxis])

        # Determine closest crossed boundary (if any)
        dist_from_start = np.abs(boundaries - z0[..., None])
        dist_from_start[~crossed_mask] = np.inf  # Ignore non-crossed boundaries

        closest_idx = np.argmin(dist_from_start, axis=-1)

        plane = np.where(np.any(crossed_mask, axis=-1), boundaries[closest_idx], None).astype(np.float64)

        # Determine mediums at the interface
        has_interface = np.any(crossed_mask, axis=-1)
        if np.any(has_interface):
            interface0 = np.where(has_interface, self.in_medium(np.nextafter(plane, z0)), None)
            interface1 = np.where(has_interface, self.in_medium(np.nextafter(plane, z1)), None)

            # Combine mediums into tuples or return empty tuples where no interface was crossed
            interfaces = np.array(
                [tuple(pair) if valid else () for valid, pair in zip(has_interface, zip(interface0, interface1))],
                dtype=object)
            return interfaces, plane
        else:
            return None, None

    def represent_on_axis(self,
                          ax: plt.axes.Axes = None) -> None:
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
            alpha = 0
            for bound, medium in self.stack.items():
                depth = np.diff(bound)
                y_edge = bound[0] - 0.1 * depth
                x_edge = ax.get_xlim()[0] * 0.95
                ax.text(x_edge, y_edge, medium.desc, fontsize=12)
                line_x = 100 * np.asarray(ax.get_xlim())
                y1 = bound[0] if not np.isinf(bound[0]) else ax.get_ylim()[0]
                y2 = bound[1] if not np.isinf(bound[1]) else ax.get_ylim()[1]
                alpha += 0.2
                ax.fill_between(line_x, y1, y2,
                                color='gray' if medium.display_color is None else medium.display_color,
                                alpha=alpha if medium.display_color is None else 1)


class IndexableProperty(np.ndarray):
    def __new__(cls,
                arr: Iterable,
                normalize: bool = False,
                dtype: Optional[np.dtype] = None) -> IndexableProperty:
        obj = np.asarray(arr, dtype=dtype).view(cls)
        obj.normalize = normalize
        obj /= np.linalg.norm(obj, axis=-1)[..., np.newaxis] if normalize else 1
        return obj

    def __array_finalize__(self, obj: Optional[NDArray]):
        if obj is None:
            return None
        self._normalize = getattr(obj, '_normalize', False)

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value
        if self._normalize:
            self /= np.linalg.norm(self, axis=-1)[..., np.newaxis]

    def __setitem__(self,
                    index: int,
                    value: Real):
        super().__setitem__(index, value)
        if self.normalize:
            self /= np.linalg.norm(self, axis=-1)[..., np.newaxis]

    def __getitem__(self, index: int) -> IndexableProperty[Real]:
        item = super().__getitem__(index)
        if isinstance(item, IndexableProperty):
            item.normalize = False
        return item


class Photon:
    class Photon:
        """
        Represents a photon or a batch of photons with properties and behaviors that simulate photon interactions
        with a medium, including movement, absorption, scattering, and more.

        This class simulates the behavior of photons in an optical system, including their movement through
        different mediums, interactions with interfaces, and absorption or scattering events. The photons are
        modeled as a batch, and each photon has attributes such as wavelength, directional cosines, location,
        and weights, which can be modified and tracked throughout the simulation.

        Attributes:
        - batch_size (int): The number of photons in the batch.
        - wavelength (Union[Real, Iterable[Real]]): The wavelength(s) of the photon(s).
        - system (Optional[System]): The system that contains the photon batch.
        - directional_cosines (IndexableProperty[Real]): The directional cosines for each photon in the batch.
        - location_coordinates (np.ndarray): The coordinates of the location of each photon in the batch.
        - weights (np.ndarray): The weights of each photon in the batch.
        - russian_roulette_constant (Real): The constant used for photon survival in the Russian roulette process.
        - recurse (bool): Whether recursion is allowed in the simulation.
        - recursion_depth (int): The current depth of recursion in the simulation.
        - recursion_limit (Real): The limit for recursion depth in the simulation.
        - throw_recursion_error (bool): Whether to throw an error when recursion limit is exceeded (alt. warn and break).
        - keep_secondary_photons (bool): Whether to retain secondary (recursed) photons in the simulation.
        - tir_limit (Real): The limit for total internal reflection (TIR).
        - A (Real): The amount of the batch that has been absorbed.
        - R (Real): The amount of the batch that has been reflected (out of the system).
        - T (Real): The amount of the batch that has been transmitted (out of the system).
        - exit_location (np.ndarray[Real]): The coordinates of the photon at the end of the simulation.
        - exit_direciton (np.ndarray[Real]): The direction of the photon at the end of the simulation.
        - exit_weights (np.ndarray[Real]): The weights of the photon at the end of the simulation.
        - location_history (np.ndarray[Real]): The location history of the photon at all steps in the batch simulation.
        - weights_history (np.ndarray[Real]): The weights history of the photon at all steps in the batch simulation.
        - cache_register (np.ndarray[np.bool_]): A cache register to track whether medium needs to be re-queried returned from the cache.
        - at_interface (np.ndarray[np.bool_]): A boolean array indicating whether the photon is at an interface currently.


        Methods:
        - __init__: Constructor for the photon that handles implicit batching necessities.
        - simulate: Runs the photon simulation until all photons are terminated.
        - absorb: Decrements each photon's weights based on current medium's albedo.
        - move: Moves the photons one step, handling interface crossing internally.
        - reflect_refract: Deterministically updates the direciton of photons based on interface properties.
        - scatter: Randomly updates the direction of each photon based on the scattering properties of the medium.
        - copy: Creates a deep copy of the photon object, with optional attribute overwrites.
        - russian_roulette: Simulates photon survival using the Russian roulette method.
        - __repr__: Provides a string representation of the photon object for debugging.
        - plot_path: A plotting function to plot the photon histories in 3d or projected onto an input plane.
        """

    def __init__(self, wavelength: Union[Real, Iterable[Real]],
                 batch_size: int = 0,
                 system: Optional[System] = None,
                 directional_cosines: Iterable[Real] = (0, 0, 1),
                 location_coordinates: Iterable[Real] = (0, 0, 0),
                 weights: Union[Real, Iterable[Real]] = 1,
                 russian_roulette_constant: Real = 20,
                 recurse: bool = True,
                 recursion_depth: Optional[int] = 0,
                 recursion_limit: Optional[Real] = 100,
                 throw_recursion_error: bool = True,
                 keep_secondary_photons: bool = False,
                 tir_limit: Optional[Real] = float('inf')) -> None:
        """
           Initializes a photon batch with specified attributes and sets up trackers for simulation.

           The constructor ensures that directional cosines, location coordinates, and weights are appropriately sized
           to match the batch size (`n`). If only one set is provided, it will be repeated to fill the batch.
           If multiple sets are provided, they must match the batch size.

           Parameters:
           :param wavelength: A single wavelength or an iterable of wavelengths for the photon batch.
           :type wavelength: Union[Real, Iterable[Real]]
           :param n: The number of photons in the batch. Defaults to 0.
           :type n: int, optional
           :param system: The system containing the photon batch. Defaults to None.
           :type system: Optional[System], optional
           :param directional_cosines: The directional cosines of the photon batch. Can be a single set or an iterable
               matching the batch size. Defaults to (0, 0, 1).
           :type directional_cosines: Iterable[Real], optional
           :param location_coordinates: The location coordinates of the photon batch. Can be a single set or an iterable
               matching the batch size. Defaults to (0, 0, 0).
           :type location_coordinates: Iterable[Real], optional
           :param weights: The weights of the photon batch. Can be a single value or an iterable matching the batch size. Defaults to 1.
           :type weights: Union[Real, Iterable[Real]], optional
           :param russian_roulette_constant: The constant used for photon survival in the Russian roulette process. Defaults to 20.
           :type russian_roulette_constant: Real, optional
           :param recurse: A flag indicating whether recursion is allowed in the simulation. Defaults to True.
           :type recurse: bool, optional
           :param recursion_depth: The current depth of recursion. Defaults to 0.
           :type recursion_depth: int, optional
           :param recursion_limit: The limit for recursion depth. Defaults to 100.
           :type recursion_limit: Real, optional
           :param throw_recursion_error: A flag indicating whether to throw an error when the recursion limit is exceeded. Defaults to True.
           :type throw_recursion_error: bool, optional
           :param keep_secondary_photons: A flag indicating whether to retain secondary photons in the simulation. Defaults to False.
           :type keep_secondary_photons: bool, optional
           :param tir_limit: The limit for total internal reflection (TIR). Defaults to infinity.
           :type tir_limit: Real, optional

           Initializes the photon state, sets up directional cosines, location coordinates, and weights for the photon batch.
           Sets up trackers for exit direction, exit location, exit weights, and secondary photons. Initializes photon
           tracking history, such as location and weights history, as well as recursion and TIR count trackers.

           """

        # Init photon state
        self.batch_size = batch_size
        self.wavelength = wavelength
        self.system = system
        self.russian_roulette_constant = russian_roulette_constant
        self._medium = None
        self.recurse = recurse
        self.recursion_depth = recursion_depth
        self.recursion_limit = recursion_limit
        self.throw_recursion_error = throw_recursion_error
        self.keep_secondary_photons = keep_secondary_photons
        self.tir_limit = tir_limit

        # Setup batched attributes
        self._directional_cosines = IndexableProperty(self._batch_fill(directional_cosines, dtype=np.float64))
        self._location_coordinates = self._batch_fill(location_coordinates, dtype=np.float64)
        self._weights = self._batch_fill(weights)

        # Init all-false cache register (so medium will be filled in for all)
        self.cache_register = np.repeat(np.False_, self.batch_size, axis=0)

        # Init empty medium cache and at_interface cache
        self._medium = np.empty((self.batch_size,), dtype=object)
        self.at_interface = np.empty((self.batch_size,), dtype=np.bool_)

        # Exit trackers
        self.exit_direction = np.empty_like(self.directional_cosines)
        self.exit_direction[:] = np.nan
        self.exit_location = np.empty_like(self.location_coordinates)
        self.exit_location[:] = np.nan
        self.exit_weights = np.empty(self.batch_size)
        self.exit_weights[:] = np.nan

        self.secondary_photons = []

        # Call setter in case current_photon is DOA
        self.weights = weights

        # Init trackers
        self.location_history = self.location_coordinates[..., np.newaxis].copy()
        self.weights_history = self.weights[..., np.newaxis].copy()
        self.recursed_photons = np.zeros(self.batch_size, dtype=np.uint16)
        self.tir_count = np.zeros(self.batch_size, dtype=np.uint16)
        self.A = 0.
        self.T = 0.
        self.R = 0.

    def _batch_fill(self, value: Union[Iterable, NDArray], dtype: Optional[np.dtype] = None) -> NDArray:
        """
        Provides a filled-in array that matches the batch size of the photon.

        This helper method ensures a value is correctly expanded to match the batch size when necessary.

        :param value: The value to assign, either an IndexableProperty or an NDArray. This will be expanded
                      to match the batch size.
        :type value: Union[IndexableProperty, NDArray]
        :return: A numpy array with the correct batch size.
        :rtype: NDArray[np.float64]
        """
        value = np.asarray(value, dtype=dtype)
        # Return already batched inputs
        if value.shape and value.shape[0] == self.batch_size:
            return value

        # Determine singlet size of value (in cases of pre-dimed/unsqueezed arrays, shape[-1] will give singlet shape)
        base_shape = 1 if np.ndim(value) == 0 else np.shape(value)[-1]
        if np.ndim(value) <= 1:
            value = np.repeat(value[np.newaxis, ...], self.batch_size, axis=0)
        elif np.ndim(value) == 2 and np.shape(value)[0] == 1:
            value = np.repeat(value, self.batch_size, axis=0)
        else:
            raise ValueError(
                f'Input is incompatible with batch size. Input must be shape ({base_shape},) or '
                f'({self.batch_size}, {base_shape}) but the input is {np.shape(value)}'
            )
        return value

    @property
    def location_coordinates(self) -> NDArray[Real]:
        """
        Retrieves the current value of the photon location coordinates.

        :return: The current value of the location coordinates as an np.ndarray.
        :rtype: NDArray[Real]
        """
        return self._location_coordinates

    @location_coordinates.setter
    def location_coordinates(self, location_coordinates: Iterable[Real]) -> None:
        """
        Sets the current value of the location coordinates and update the medium cache register for those that have 
        change.
        :param location_coordinates: 
        :return: 
        """

        # Ensure ndarray and fill to batch size
        location_coordinates = self._batch_fill(np.asarray(location_coordinates, dtype=np.float64))

        # Determine change status
        unchanged = (location_coordinates[:, 2] == self.location_coordinates[:, 2])

        # Set coordinates
        self._location_coordinates = location_coordinates

        # Update cache register
        self.cache_register = np.where(unchanged, self.cache_register, np.False_)

    @property
    def directional_cosines(self) -> IndexableProperty[Real]:
        """
        Retrieves the current value of the photon directional cosines.

        :return: The current value of the directional cosines, wrapped in an `IndexableProperty` object.
        :rtype: IndexableProperty[Real]
        """
        return self._directional_cosines

    @directional_cosines.setter
    def directional_cosines(self, directional_cosines: Iterable[Real]) -> None:
        """
        Setter for the photon directional cosines that ensures the normalization is maintained and updated the cache 
        register where the sign of the z-direction changes to trigger medium updates for those photons.

        This setter automatically re-assigns the updated value to an indexable property object,
        which handles normalization both at creation and when setting values using an indexed `__setitem__`.

        :param value: An iterable of real numbers representing the directional cosines of the photon.
        :type value: Iterable[Real]
        :return: This method updates the directional cosines and does not return any value.
        :rtype: None
        """
        # Ensure Indexable Property and fill to batch size
        directional_cosines = IndexableProperty(self._batch_fill(directional_cosines, dtype=np.float64), normalize=True)

        # Determine changed directions
        unchanged = (np.sign(directional_cosines[..., 2]) == np.sign(self.directional_cosines[:, 2]))

        # Set directional cosines
        self._directional_cosines = directional_cosines

        # Update cache register
        self.cache_register = np.where(unchanged, self.cache_register, np.False_)

    def copy(self, **kwargs: Any) -> Photon:
        """
        Creates a deep copy of the Photon object and allows for overwriting specific attributes using **kwargs.

        This method creates a new instance of the Photon object with the same attributes as the original. It can also
        overwrite the values of specified attributes using the keyword arguments passed to it. The tracker attributes
        (`T`, `R`, `A`, `tir_count`, `recursed_photons`) are automatically reset to 0 in all cases.

        :param kwargs: Keyword arguments representing attributes to overwrite in the copied object.
        :type kwargs: dict
        :return: A new Photon object that is a deep copy of the original, with overwritten attributes if provided.
        :rtype: Photon
        """

        new_obj = copy.deepcopy(self)

        # Check for kwarg overwrites
        for key, value in kwargs.items():
            if hasattr(new_obj, key):
                setattr(new_obj, key, value)
            elif hasattr(new_obj, f'_{key}'):
                setattr(new_obj, f'_{key}', value)

        # Reset tracker attributes
        for key in ['T', 'R', 'A']:
            setattr(new_obj, key, 0)
        for key in ['tir_count', 'recursed_photons']:
            setattr(new_obj, key, np.zeros(self.batch_size, dtype=np.uint8))
        new_obj.location_history = self.location_coordinates[..., np.newaxis].copy()
        new_obj.weights_history = self.weights[..., np.newaxis].copy()

        return new_obj

    def __repr__(self) -> str:
        """
        Returns a string representation of the Photon object for quick debugging or analysis.

        This method generates a human-readable string that includes the object's attributes and their values,
        providing an easy way to inspect the state of the Photon.

        :return: A string representation of the Photon object.
        :rtype: str
        """
        out = ''
        for key, val in self.__dict__.items():
            out += f"{key.strip('_')}: {val}\n"
        return out

    def simulate(self) -> None:
        """
        Simulates the behavior of a batch of photons until all photons are terminated and updates the object's
        attributes accordingly.

        The simulation involves the following steps:
        - **Absorption**: Photons interact with the medium and may be absorbed. (See `self.absorb`).
        - **Movement**: Photons move within the medium. (See `self.move`).
        - **Scattering**: Photons may scatter as they travel through the medium. (See `self.scatter`).

        The exact behavior of these events depends on the optical properties of the `Medium` object the photon is in,
        which is determined through queries to the `System` that contains the photon, i.e., `self.system`.

        :return: This method runs the simulation and does not return any value.
        :rtype: None
        """

        if not self.system is not None:
            raise RuntimeError('Photon must be in an Optical System object to simulate.')
        if self.recursion_depth >= self.recursion_limit:
            raise RecursionError(
                'Maximum photon recursion limit reached. Recursion depth limit can be increased with the '
                'recursion_limit attribute.\n'
                'To switch this error off and throw a warning instead, set throw_recursion_error to FALSE. This '
                'will simulate the photon to the limit without recursion, rather than throwing an error.')
        while not self.is_terminated:
            self.absorb()
            self.move()
            self.scatter()

    @property
    def weights(self) -> NDArray[Real]:
        """
        Retrieves the current weights of the photons.
        Retrieves the current weights of the photons.

        :return: An array of real values representing the photon weights.
        :rtype: NDArray[Real]
        """
        return self._weights

    @weights.setter
    def weights(self, weights: Union[Real, Iterable[Real]]) -> None:
        """
        Sets the photon weights and applies Russian roulette if the weights falls below the threshold of 0.005.

        If the weights is below 0.005, the `russian_roulette` method is called to determine whether the photon
        survives or is terminated.

        :param weights: The new weights value(s) to set. Can be a single real number or an iterable of real numbers.
        :type weights: Union[Real, Iterable[Real]]
        :return: This method updates the weights in place and does not return a value.
        :rtype: None
        """
        if isinstance(weights, Real) or np.shape(weights)[0] == 1:
            weights *= np.ones(self.batch_size)
        self._weights = np.where(weights > 0, weights, 0)
        rr_check = (0 < weights) & (weights < 0.005)
        if np.any(rr_check):
            self.russian_roulette(rr_check)

    def russian_roulette(self, mask: NDArray[np.bool_]) -> None:
        """
        Determines photon survival using the Russian roulette technique.

        If a photon survives, its weights is increased by `self.russian_roulette_constant`.
        If it does not survive, its weights is set to `0`, effectively terminating it.

        :param mask: A boolean array where `True` indicates that the corresponding photon is
                     subject to the Russian roulette survival test.
        :type mask: NDArray[np.bool_]
        :return: This method modifies photon weights in place and does not return a value.
        :rtype: None
        """
        survival = np.random.rand(np.count_nonzero(mask)) < (1 / self.russian_roulette_constant)
        self._weights[mask] = np.where(survival, self._weights[mask] * self.russian_roulette_constant, 0)

    @property
    def medium(self) -> NDArray[Medium]:
        """
        Determines the current medium for the photons. If the current medium is cached, it is returned from the cache,
        otherwise it is updated by querying the system with the location.

        If the photons are at an interface, the function returns the medium the photon is moving into,
        using `headed_into`. Otherwise, it returns the current medium.

        :return: An array of `Medium` objects representing the current medium for the photons.
        :rtype: NDArray[Medium]
        """
        # Fill in from the cache
        cached = self._medium
        self._medium = np.empty((self.batch_size,), dtype=object)
        self._medium[self.cache_register] = cached[self.cache_register]

        # Update changed photons
        self._medium[~self.cache_register] = self.system.in_medium(self.location_coordinates[~self.cache_register])

        # Photons that are at an interface have a tuple returned from system query
        self.at_interface[~self.cache_register] = \
            np.array([isinstance(medium, (tuple, list)) for medium in self._medium])[~self.cache_register]

        # Get the headed_into medium for those photons
        if np.any(self.at_interface):
            self._medium[self.at_interface] = self.headed_into(mediums=self._medium)[self.at_interface]

        # Update the cache register to all-true
        self.cache_register[:] = np.True_

        return self._medium

    @property
    def is_terminated(self) -> np.bool_:
        """
        Checks if there are still photons in the batch to simulate.

        This property returns `True` if all photons have been terminated (i.e., they have exited the simulation
        or met termination criteria). Otherwise, it returns `False`.

        :return: A boolean value indicating whether the simulation has completed for all photons.
        :rtype: np.bool_
        """

        self._is_terminated = np.all((self.medium == self.system.surroundings) | (self.weights <= 0.0))
        return self._is_terminated

    def headed_into(self, mediums: Optional[NDArray[Union[Medium, Tuple[Medium, Medium]]]] = None) -> NDArray[Medium]:
        """
        Determines which medium a photon is headed into, particularly when at an interface.

        If `mediums` is provided, the function uses it instead of re-querying `self.system.in_mediums` to determine the
        current state. In cases where `mediums` contains tuples `(Medium, Medium)`, the function selects the appropriate
        medium based on direction:
        - The first element (index `0`) represents the negative-direction medium.
        - The second element (index `1`) represents the positive-direction medium.

        If `mediums` is not a tuple, the function simply returns the same medium.

        :param mediums: An optional array of mediums. Each element can be:
            - A `Medium` object, indicating a single medium.
            - A tuple `(Medium, Medium)`, representing a negative and positive medium pair at an interface.
            If not provided, the function queries `self.system.in_mediums` instead.
        :type mediums: NDArray[Union[Medium, Tuple[Medium, Medium]]], optional
        :return: An array of `Medium` objects representing the medium the photon is headed into.
        :rtype: NDArray[Medium]
        """

        mediums = self.system.in_medium(self.location_coordinates) if mediums is None else mediums
        in_mask = np.array([isinstance(medium, (tuple, list)) for medium in mediums])
        neg_medium = [medium[0] if in_ else np.nan for in_, medium in zip(in_mask, mediums)]
        pos_medium = [medium[1] if in_ else np.nan for in_, medium in zip(in_mask, mediums)]
        headed_into = np.where(
            ~in_mask,
            mediums,
            np.where(
                self.directional_cosines[:, 2] < 0,
                neg_medium,
                pos_medium
            )
        )

        return headed_into

    def absorb(self) -> None:
        """
        Decrements the weights of the photon batch according to the albedo of the current medium.

        This method simulates the absorption of photons in the current medium. The weights of each photon is
        reduced based on the albedo, which represents the fraction of light that is reflected. The remaining
        weights is used to continue the photon simulation.

        The specific absorption process depends on the optical properties of the medium the photon is currently in.

        :return: None. The photon batch's weights are updated in place.
        :rtype: None
        """

        absorbed_weights = self.weights * np.array([medium.albedo_at(self.wavelength) for medium in self.medium])
        self.A += np.sum(absorbed_weights)
        self.weights = self.weights - absorbed_weights

    def move(self, step: Union[Real, Iterable[Real]] = None) -> None:
        """
        Moves the photon one step in its current direction.

        The size of the step can be either provided directly, or it can be determined by the mean free path of the
        current medium, which is calculated from the transport coefficient (mu_t). If the photon is in a medium with
        mu_t = 0 (i.e., no scattering or absorption), the photon will automatically advance to the next interface along
        its direction.

        If a step size is provided, the photon moves in the direction specified by its directional cosines until it
        either completes the step or hits an interface. If the full step would cross an interface, only the portion of
        the step before the interface is executed. At the interface, the photon is refracted and reflected according to
        the media properties, and its weights is updated (decremented) based on the interaction.

        :param step: The distance the photon should move. If not provided, the step size is sampled based on the mean
        free path.
        :type step: Union[Real, Iterable[Real]], optional
        :return: None. The photon’s state (location, weights) is updated in place.
        :rtype: None
        """

        # Get current state
        mu_t = np.array([medium.mu_t_at(self.wavelength) for medium in self.medium])
        dir_cos = self.directional_cosines
        loc = self.location_coordinates

        # If scattering occurs, step is sampled from the distribution
        if step is None:
            step = np.where(mu_t > 0, -np.log(np.random.rand(self.batch_size)) / mu_t, float('inf'))
            step = np.where(self.weights > 0, step, 0)
        step = self._batch_fill(step)
        new_loc = loc + step[:, np.newaxis] * dir_cos

        # Determine which photons cross an interfaces
        interface, plane = self.system.interface_crossed(loc, new_loc)
        crossed = (~np.isnan(plane)) if plane is not None else False
        move_to_interface = False
        if np.any(crossed):
            interface_steps = np.where(crossed, (plane - loc[:, 2]) / dir_cos[:, 2], float('inf'))

            # Find photons that should move to the interfaces instead
            move_to_interface = (interface_steps < step) & crossed
            new_loc[move_to_interface] = (loc[move_to_interface] + interface_steps[move_to_interface, np.newaxis]
                                          * dir_cos[move_to_interface])

        # Update location
        self.location_coordinates = new_loc

        # Reflect/refract photons at interfaces
        if np.any(move_to_interface):
            self.reflect_refract(interface, move_to_interface)

        # Update history for all photons
        self.location_history = np.append(self.location_history, new_loc[..., np.newaxis], axis=2)
        self.weights_history = np.append(self.weights_history, self.weights[..., np.newaxis], axis=1)

        # Check if any new photons exited
        exit_mask = (self.headed_into() == self.system.surroundings) & (self.weights > 0)

        if np.any(exit_mask):
            self.exit_location[exit_mask] = self.location_history[exit_mask, ..., -1]
            self.exit_weights[exit_mask] = self.weights_history[exit_mask, -1]

            # Check if any exited photons hit a detector
            detector_mask = exit_mask & (self.exit_location[:, 2] == self.system.detector_location)
            if np.any(detector_mask):
                self.system.detector(self, exit_mask)

            # Handle reflection or transmission
            self.R += np.sum(np.where(exit_mask & (self.directional_cosines[:, 2] < 0), self.weights, 0))
            self.T += np.sum(np.where(exit_mask & (self.directional_cosines[:, 2] > 0), self.weights, 0))

            # Terminate exited photons
            self.weights[exit_mask] = 0

    def reflect_refract(self,
                        interfaces: NDArray[object[Tuple[Medium, Medium], Tuple[None]]],
                        mask: NDArray[np.bool_]) -> None:
        """
        Computes photon reflection and refraction at interfaces.

        This method determines the weight of photon reflection and updates the tracker's direction accordingly.
        If reflection occurs, it either updates the photon direction or spawns secondary photons with adjusted weights
        to continue the recursive simulation. The photon's direction is updated based on the interface's refractive
        indices. This operation is performed only on photons specified by the input mask.

        :param interfaces: NumPy array containing interface definitions.
                           Each element is a tuple of (Medium, Medium) for a valid interface or (None,) if no interface
                           is present.
        :type interfaces: np.ndarray[object]
        :param mask: Boolean mask indicating which photons to process.
        :type mask: NDArray[np.bool_]
        :return: None. The photon's direction and weight are updated along with relevant trackers,
                 and secondary photons are spawned if necessary.
        """

        # Get incidence state
        mu_x, mu_y, mu_z_i = self.directional_cosines[mask].T
        mu_z_t = np.zeros_like(mu_z_i)
        n1 = np.array(
            [interface[0].n if iterable(interface) and len(interface) == 2 else np.nan for interface in
             interfaces])[mask]
        n2 = np.array(
            [interface[1].n if iterable(interface) and len(interface) == 2 else np.nan for interface in
             interfaces])[mask]

        # Calculate refraction
        sin_theta_t = n1 / n2 * np.sqrt(1 - (mu_z_i ** 2))

        # TIR
        tir_mask = sin_theta_t > 1
        mu_z_t[tir_mask] = -mu_z_i[tir_mask]
        self.tir_count[mask] = np.where(tir_mask, self.tir_count[mask] + 1, self.tir_count[mask])
        stop_tir = self.tir_count[mask] > self.tir_limit
        self.A += np.sum(np.where(stop_tir, self.weights[mask], 0))
        self.weights[mask] = np.where(stop_tir, 0, self.weights[mask])

        # Snell's + Fresnel's Law
        refract_mask = ~tir_mask
        mu_z_t_masked = np.sqrt(1 - (sin_theta_t[refract_mask] ** 2))

        # Extract only the masked refractive indices for masked values
        n1_masked = n1[refract_mask]
        n2_masked = n2[refract_mask]
        abs_mu_z_i = np.abs(mu_z_i[refract_mask])

        rs = np.abs(((n1_masked * abs_mu_z_i) - (n2_masked * mu_z_t_masked)) /
                    ((n1_masked * abs_mu_z_i) + (n2_masked * mu_z_t_masked))) ** 2
        rp = np.abs(((n2_masked * mu_z_t_masked) - (n1_masked * abs_mu_z_i)) /
                    ((n1_masked * abs_mu_z_i) + (n2_masked * mu_z_t_masked))) ** 2
        specular_reflection = 0.5 * (rs + rp) * self.weights[mask][refract_mask]

        mu_z_t[refract_mask] = mu_z_t_masked * np.sign(mu_z_i[refract_mask])  # Ensure correct sign

        # Updated for transmitted portion
        mu_x[refract_mask] *= n1_masked / n2_masked
        mu_y[refract_mask] *= n1_masked / n2_masked

        if self.recurse:
            # Setup new photon attributes
            batch_size = np.sum(refract_mask).item()
            dir_cos = np.array([[1, 1, -1]]) * self.directional_cosines[mask][refract_mask]  # Flipped for reflection
            weights = specular_reflection
            loc_cor = self.location_coordinates[mask][refract_mask]
            rec_dep = self.recursion_depth + 1
            secondary_photons = Photon(self.wavelength,
                                       system=self.system,
                                       batch_size=batch_size,
                                       location_coordinates=loc_cor,
                                       directional_cosines=dir_cos,
                                       weights=weights,
                                       recursion_depth=rec_dep)
            secondary_photons._medium = self._medium[mask][refract_mask]
            secondary_photons.cache_register = self.cache_register[mask][refract_mask]
            secondary_photons.at_interface = self.at_interface[mask][refract_mask]
            try:
                secondary_photons.simulate()
            except RecursionError as e:
                if self.throw_recursion_error:
                    raise e
                else:
                    warnings.warn(str(e), RuntimeWarning)
            finally:
                self.R += secondary_photons.R
                self.T += secondary_photons.T
                self.A += secondary_photons.A
                if self.keep_secondary_photons:
                    self.secondary_photons.append(secondary_photons)
                    self.secondary_photons += secondary_photons.secondary_photons

        else:
            # If the reflected fraction will be reflected out, add it to reflected count, Else add it to transmitted
            reflected_out = mu_z_i[refract_mask] > 0
            transmitted_out = mu_z_i[refract_mask] < 0
            self.R += np.sum(specular_reflection * reflected_out)
            self.T += np.sum(specular_reflection * transmitted_out)

        w_temp = self.weights[mask].copy()
        w_temp[refract_mask] -= specular_reflection
        self.weights[mask] = w_temp

        # Send to setter for normalization
        self.directional_cosines[mask] = np.column_stack((mu_x, mu_y, mu_z_t))

    def scatter(self, theta_phi: Optional[Union[Iterable[Real, Real], Iterable[Iterable[Real, Real]]]] = None):
        """
        This method updates the direciton of the photon members where they are in scattering media, but not at an
        interface.

        :param theta_phi: Angles to udpte direction with. Optional.
        :type theta_phi: Union[Iterable[Real, Real], Iterable[Iterable[Real, Real]]]
        :return: None. Updates photon directions where the photon is in scattering media (but not at an interface)
        """
        # Early break if all are at an interface or in non-scattering medium
        g = np.array([medium.g for medium in self.medium])  # (also forces eset of interface cache where necessary)
        if np.all(self.at_interface) or np.all(g == 1):
            return

        # Placeholders for angle samples
        theta = np.zeros(self.batch_size, dtype=np.float64)
        cosine_theta = np.zeros_like(theta, dtype=np.float64)
        if theta_phi is None:
            # Sample random scattering angles from distribution
            [xi, zeta] = np.random.rand(self.batch_size, 2).T

            # For non-zero g
            non_zero_g_mask = g != 0
            lead_coeff = 1 / (2 * g[non_zero_g_mask])
            term_2 = g[non_zero_g_mask] ** 2
            term_3 = (1 - term_2) / (1 - g[non_zero_g_mask] + (2 * g[non_zero_g_mask] * xi[non_zero_g_mask]))
            cosine_theta[non_zero_g_mask] = lead_coeff * (1 + term_2 - term_3)

            # For g=0
            cosine_theta[~non_zero_g_mask] = 1 - (2 * xi[~non_zero_g_mask])

            theta = np.arccos(cosine_theta)
            phi = 2 * np.pi * zeta
        else:
            theta, phi = theta_phi if isinstance(theta_phi, (tuple, list)) else zip(theta_phi)
        # Update direction cosines
        mu_x, mu_y, mu_z = self.directional_cosines.T
        new_directional_cosines = np.zeros((self.batch_size, 3), dtype=np.float64)

        # For near-vertical photons (simplify for stability)
        vertical = np.abs(mu_z) > 0.999
        new_directional_cosines[vertical, 0] = np.sin(theta[vertical]) * np.cos(phi[vertical])
        new_directional_cosines[vertical, 1] = np.sign(mu_z[vertical]) * np.sin(theta[vertical]) * np.sin(phi[vertical])
        new_directional_cosines[vertical, 2] = np.sign(mu_z[vertical]) * np.cos(theta[vertical])

        # For all others
        nonvertical = ~vertical
        deno = np.sqrt(1 - (mu_z[nonvertical] ** 2))

        # mu_x updated
        numr = (mu_x[nonvertical] * mu_z[nonvertical] * np.cos(phi[nonvertical])) - (
                mu_y[nonvertical] * np.sin(phi[nonvertical]))

        new_directional_cosines[nonvertical, 0] = (np.sin(theta[nonvertical]) * (numr / deno)) + (
                mu_x[nonvertical] * np.cos(theta[nonvertical]))

        # mu_y update
        numr = (mu_y[nonvertical] * mu_z[nonvertical] * np.cos(phi[nonvertical])) + (
                mu_x[nonvertical] * np.sin(phi[nonvertical]))

        new_directional_cosines[nonvertical, 1] = (np.sin(theta[nonvertical]) * (numr / deno)) + (
                mu_y[nonvertical] * np.cos(theta[nonvertical]))

        # mu_z update
        new_directional_cosines[nonvertical, 2] = -(np.sin(theta[nonvertical]) * np.cos(phi[nonvertical]) * deno) + (
                mu_z[nonvertical] * np.cos(theta[nonvertical]))

        # Update directional cosines with new direciton (done at once for normalization consistency)
        self.directional_cosines[~self.at_interface] = new_directional_cosines[~self.at_interface]

    def plot_path(
            self,
            project_onto: Optional[str] = None,
            axes: Optional[plt.Axes] = None,
            ignore_outside: bool = True
    ) -> None:
        """
        Visualizes the photon location histories as a 2D or 3D plot.

        - If `project_onto` is specified, the path is projected onto a 2D plane.
          Accepted values: 'xy', 'xz', 'yz'.
        - If `project_onto` is None, a 3D plot is generated.
        - If `axes` is provided, the plot is drawn on the given `matplotlib.Axes` object.
          Otherwise, a new figure and axes are created.
        - If `ignore_outside` is True (default), only locations within the system are plotted,
          and steps after exit are truncated.

        :param project_onto: Plane onto which the path is projected ('xy', 'xz', 'yz'), or None for 3D.
        :type project_onto: Optional[str]
        :param axes: Matplotlib Axes object for plotting, or None to create new axes.
        :type axes: Optional[plt.Axes]
        :param ignore_outside: Whether to ignore steps after exiting the system (default: True).
        :type ignore_outside: bool
        :return: None
        """

        project_onto = ['xz', 'yz', 'xy'] if project_onto == 'all' else project_onto
        project_onto = [project_onto] if isinstance(project_onto, str) or project_onto is None else project_onto
        batch_size, _, steps = self.location_history.shape  # Expect shape (batch, 3, steps)

        # Boundaries for filtering
        z_min, z_max = self.system.boundaries[0], self.system.boundaries[-1]
        inside = ((self.location_history[:, 2] >= z_min)
                  & (self.location_history[:, 2] <= z_max)) if ignore_outside else True

        fig = plt.figure(figsize=(8 * len(project_onto), 8)) if not plt.get_fignums() else plt.gcf()
        if project_onto[0]:
            axes = [fig.add_subplot(1, len(project_onto), i + 1) for i in
                    range(len(project_onto))] if axes is None else axes
            for ax, projection in zip(axes, project_onto):
                for i in range(batch_size):
                    x, y = self.location_history[i, 'xyz'.index(projection[0])], self.location_history[
                        i, 'xyz'.index(projection[1])]
                    ax.plot(x[inside[i]], y[inside[i]], label=f'Photon {i + 1}')
                ax.set_title(f'Projected onto {projection}-plane')
                ax.set_xlabel(f'Photon Displacement in {projection[0]}-direction (cm)')
                ax.set_ylabel(f'Photon Displacement in {projection[1]}-direction (cm)')
                if projection[1] == 'z' and not ax.yaxis_inverted():
                    ax.invert_yaxis()
                if projection[0] == 'z' and not ax.xaxis_inverted():
                    ax.invert_xaxis()
        else:
            axes = fig.add_subplot(projection='3d') if axes is None else axes
            for i in range(batch_size):
                x, y, z = self.location_history[i, 0], self.location_history[i, 1], self.location_history[i, 2]
                axes.plot(x[inside[i]], y[inside[i]], z[inside[i]], label=f'Photon {i + 1}')
            axes.set_title('Photon Paths')
            axes.set_xlabel('Photon Displacement in x-direction (cm)')
            axes.set_ylabel('Photon Displacement in y-direction (cm)')
            axes.set_zlabel('Photon Displacement in z-direction (cm)')
            if not axes.zaxis_inverted():
                axes.invert_zaxis()

        return fig, axes
