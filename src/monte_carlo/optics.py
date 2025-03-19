from __future__ import annotations

import copy
import warnings
from numbers import Real
from typing import Union, Optional, Iterable, Tuple, Callable, List

from cupy.typing import NDArray
from matplotlib import pyplot as plt, animation

from .hardware import ring_pattern, cone_of_acceptance, ID, OD, THETA
from .utils import calculate_mus

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


class Medium:
    def __init__(self,
                 n: Real = 1,
                 mu_s: Union[Real, Iterable[Real]] = mu_s,
                 mu_a: Union[Real, Iterable[Real]] = mu_a,
                 wavelengths: Iterable[Real] = wl,
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

    def __repr__(self):
        return self.desc.capitalize() + ' Optical Medium Object'

    def __str__(self):
        if self.desc == 'default':
            return f'Optical Medium: n={self.n}, mu_s={self.mu_s}, mu_a={self.mu_a}, g={self.g}'
        return self.desc.capitalize()

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
        return mu_s

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

    def photon(self, n: int = 50000, **kwargs) -> Photon:
        location, direction = self.pattern(n)
        wavelength = sample_spectrum(self.spectrum) if self.spectrum else None
        return Photon(wavelength, n=n, location_coordinates=location, directional_cosines=direction, **kwargs)


class Detector:
    def __init__(self, acceptor: Callable = cone_of_acceptance(ID), desc: Optional[str] = 'default') -> None:
        self.acceptor = acceptor
        self.n_total = 0
        self.n_detected = 0
        self.desc = desc

    def detect(self, location: Iterable[Real],
               direction: Union[Real, Iterable[Real]],
               weight: Optional[Union[Real, Iterable[Real]]] = None) -> None:
        weight = weight if weight is not None else 1
        self.n_total += np.nansum(weight)
        x, y = location[:, :2].T
        mu_z = direction[:, -1]
        accepted_mask = self.acceptor(x, y, mu_z=mu_z)
        self.n_detected += np.nansum(weight[accepted_mask])

    def __call__(self,
                 photon: Photon,
                 mask: Optional[NDArray[np.bool_]] = True) -> None:
        assert isinstance(photon, Photon), ValueError('Detector object can only be called directly with a photon. '
                                                      'Use detector.detect() for non-photon test cases.')
        self.detect(photon.exit_location[mask], photon.exit_direction[mask], photon.exit_weight[mask])

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
        self._boundaries = []

        assert len(args) % 2 == 0, "Arguments must be in groups of 2: medium object and thickness."
        self.surroundings = Medium(n=surrounding_n, mu_s=0, mu_a=0, desc='surroundings')

        # Add surroundings from -inf to 0
        interface = 0  # Current interfaces location during stacking
        self.stack = {(float('-inf'), interface): self.surroundings}  # Dict with tuple layer boundaries: layer object
        self.layer = [self.surroundings]  # List of layers in order of addition

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
        return ''.join([f' <-- {bound[0]} --> {layer}' for bound, layer in self.stack.items()])

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

    def add(self, layer: Medium, depth: Real) -> None:
        assert self.layer[-1] == self.surroundings or self.boundaries[-1] != float('inf'), OverflowError('Cannot add to semi-infinite system.')
        depth = float(depth)
        # Get the bound of the last layer before surroundings
        for bound, medium in self.stack.items():
            if medium == self.surroundings and bound[0] != float('inf'):
                interface = bound[0] if bound[0] != float('-inf') else bound[1]
                break

        self.stack[(interface, interface + depth)] = layer
        self._boundaries.append(interface + depth)
        self.layer.append(layer)

        # Add surroundings if not semi-infinte
        if interface + depth < float('inf'):
            self.add(self.surroundings, float('inf'))

    def beam(self, n: int = 50000, **kwargs) -> Photon:
        photon = self.illuminator.photon(n=n, system=self)
        for key, val in kwargs.items():
            setattr(photon, key, val)
        return photon

    def in_medium(self, location: Iterable[Real]) -> NDArray[Union[Medium, Tuple[Medium, Medium]]]:
        """
        Return the medium(s) that are at the queried coordinate. If the coordinate is an interfaces location, the mediums
        that makeup the interfaces are returned as a tuple, this includes boundary interfaces being returned with False
        on the surroundings side.

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
        crossed_mask = (z_sorted[..., 0, None] < boundaries) & (boundaries < z_sorted[..., 1, None])

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
            alpha = 0.0
            for bound, medium in self.stack.items():
                depth = np.diff(bound)
                y_edge = bound[0] - 0.1 * depth
                x_edge = ax.set_xlim(ax.get_xlim())[0] * 0.95
                ax.text(x_edge, y_edge, medium.desc, fontsize=12)
                line_x = 100 * np.asarray(ax.get_xlim())
                alpha += 0.2
                ax.fill_between(line_x, bound[0], bound[1],
                                color='gray' if medium.display_color is None else medium.display_color,
                                alpha=alpha if medium.display_color is None else 1)


class IndexableProperty(np.ndarray):
    def __new__(cls,
                arr: Iterable,
                normalize: bool = False) -> IndexableProperty:
        obj = np.asarray(arr, dtype=np.float64).view(cls)
        obj.normalize = normalize
        obj /= np.linalg.norm(obj, axis=1)[:, np.newaxis] if normalize else 1
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
            self /= np.linalg.norm(self, axis=1)[:, np.newaxis]

    def __setitem__(self,
                    index: int,
                    value: Real):
        super().__setitem__(index, value)
        if self.normalize:
            self /= np.linalg.norm(self, axis=1)[:, np.newaxis]

    def __getitem__(self, index: int) -> IndexableProperty[Real]:
        item = super().__getitem__(index)
        if isinstance(item, IndexableProperty):
            item.normalize = False
        return item


class Photon:

    def __init__(self, wavelength: Union[Real, Iterable[Real]],
                 n: int = 0,
                 system: System = None,
                 directional_cosines: Iterable[Real] = (0, 0, 1),
                 location_coordinates: Iterable[Real] = (0, 0, 0),
                 weight: Union[Real, Iterable[Real]] = 1,
                 russian_roulette_constant: Real = 20,
                 recurse: bool = True,
                 recursion_depth: int = 0,
                 recursion_limit: Real = 100,
                 throw_recursion_error: bool = True,
                 keep_secondary_photons: bool = False,
                 tir_limit: Real = float('inf')) -> None:

        # Init photon state
        self.batch_size = n
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
        directional_cosines = np.asarray(directional_cosines, dtype=np.float64)
        if np.ndim(directional_cosines) == 1:
            directional_cosines = np.repeat(directional_cosines[np.newaxis, ...], n, axis=0)
        else:
            assert np.shape(directional_cosines)[0] == n and np.shape(directional_cosines)[1] == 3, ValueError(
                f'Directional cosine input is incompatible with batch size. Input bust be shape (3,) or ({n}, 3) but '
                f'the input is {np.shape(directional_cosines)}'
            )
        self.directional_cosines = directional_cosines

        location_coordinates = np.asarray(location_coordinates, dtype=np.float64)
        if np.ndim(location_coordinates) == 1:
            location_coordinates = np.repeat(location_coordinates[np.newaxis, ...], n, axis=0)
        else:
            assert np.shape(location_coordinates)[0] == n and np.shape(location_coordinates)[1] == 3, ValueError(
                f'Directional cosine input is incompatible with batch size. Input bust be shape (3,) or ({n}, 3) but '
                f'the input is {np.shape(location_coordinates)}'
            )
        self.location_coordinates = location_coordinates

        weight = np.asarray(weight, dtype=np.float64)
        if not iterable(weight) or np.shape(weight) == (1,):
            weight = np.repeat(weight[np.newaxis, ...], n, axis=0)
        else:
            assert np.shape(weight)[0] == n and np.shape(weight)[1] == 1, ValueError(
                f'Directional cosine input is incompatible with batch size. Input bust be shape (1,) or ({n}, 1) but '
                f'the input is {np.shape(weight)}'
            )
        self._weight = weight

        # Exit trackers
        self.exit_direction = np.empty_like(directional_cosines)
        self.exit_direction[...] = np.nan
        self.exit_location = np.empty_like(location_coordinates)
        self.exit_location[...] = np.nan
        self.exit_weight = np.empty(n)
        self.exit_weight[...] = np.nan

        self.secondary_photons = []

        # Call setter in case current_photon is DOA
        self.weight = weight

        # Init trackers
        self.location_history = self.location_coordinates[..., np.newaxis].copy()
        self.weight_history = self.weight[..., np.newaxis].copy()
        self.recursed_photons = np.zeros(n, dtype=int)
        self.tir_count = np.zeros(n, dtype=int)
        self.A = 0.
        self.T = 0.
        self.R = 0.

    # EXPLANATION OF INDEXABLE_PROPERTIES
    # When one of these properties, obj, is used in the form obj = value, it will call the setter, which sets the
    # attribute to an indexable_property object with value. When it is used in the form obj[i] = val, the getter is
    # called to return obj. Then, the indexable_property __setitem__ is called to update obj. This should ensure that
    # these properties are always np.ndarrays and (when set) are normalized.
    @property
    def directional_cosines(self) -> IndexableProperty[Real]:
        return self._directional_cosines

    @directional_cosines.setter
    def directional_cosines(self, value: Iterable[Real]) -> None:
        self._directional_cosines = IndexableProperty(value, normalize=True)

    @property
    def location_coordinates(self) -> IndexableProperty[Real]:
        return self._location_coordinates

    @location_coordinates.setter
    def location_coordinates(self, value: Iterable[Real]) -> None:
        self._location_coordinates = IndexableProperty(value)

    def copy(self, **kwargs) -> Photon:
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

    def __repr__(self) -> str:
        out = ''
        for key, val in self.__dict__.items():
            out += f"{key.strip('_')}: {val}\n"
        return out

    def simulate(self) -> None:
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
    def weight(self) -> NDArray[Real]:
        return self._weight

    @weight.setter
    def weight(self, weight: Union[Real, Iterable[Real]]) -> None:
        # If one weight is given, batch for all photons in batch
        if isinstance(weight, Real):
            weight *= np.ones(self.batch_size)
        self._weight = weight
        rr_check = (0 < weight) & (weight < 0.005)
        if np.any(rr_check):
            self.russian_roulette(rr_check)

    def russian_roulette(self, mask: NDArray[np.bool_]) -> None:
        survival = np.random.rand(np.count_nonzero(mask)) < (1 / self.russian_roulette_constant)
        self._weight[mask] = np.where(survival, self._weight[mask] * self.russian_roulette_constant, 0)

    @property
    def medium(self) -> NDArray[Medium]:
        self._medium = self.system.in_medium(self.location_coordinates)
        tuple_mask = np.array([isinstance(medium, (tuple, list)) for medium in self._medium])
        if np.any(tuple_mask):
            self._medium[tuple_mask] = self.headed_into[tuple_mask]
        return self._medium

    @property
    def is_terminated(self) -> np.bool_:
        self._is_terminated = np.all((self.medium == self.system.surroundings) | (self.weight <= 0.0))
        return self._is_terminated

    @property
    def headed_into(self) -> NDArray[Medium]:
        bumped_coords = self.location_coordinates.copy()
        medium = np.array([None] * len(bumped_coords), dtype=object)  # Placeholder for batch results

        # While some photons still have (list, tuple) as their medium, update them
        active = np.ones(len(bumped_coords), dtype=bool)

        while np.any(active):
            # Query medium for only active photons
            new_medium = np.array(self.system.in_medium(bumped_coords[active]), dtype=object)

            # Update those that have exited the list/tuple stage
            mask_done = ~np.array([isinstance(m, (list, tuple)) for m in new_medium])
            medium[active] = np.where(mask_done, new_medium, medium[active])

            # Stop processing photons that have found their medium
            active[active] = ~mask_done

            # Increment active photons by a small step in their direction
            bumped_coords[active] = np.nextafter(bumped_coords[active],
                                                 bumped_coords[active] + self.directional_cosines[active])

        return medium

    # TODO: Add support for fluorescence-based secondary photons
    def absorb(self) -> None:
        absorbed_weight = self.weight * np.array([medium.albedo_at(self.wavelength) for medium in self.medium])
        self.A += np.sum(absorbed_weight)
        self.weight = self.weight - absorbed_weight

    def move(self, step: Union[Real, Iterable[Real]] = None) -> None:
        # Get current state
        mu_t = np.array([medium.mu_t_at(self.wavelength) for medium in self.medium])
        dir_cos = self.directional_cosines
        loc = self.location_coordinates

        # If scattering occurs, step is sampled from the distribution
        if step is None:
            step = np.where(mu_t > 0, -np.log(np.random.rand(self.batch_size)) / mu_t, float('inf'))
            step = np.where(self.weight > 0, step, 0)
        new_loc = loc + step[:, np.newaxis] * dir_cos

        # Determine which photons cross an interfaces
        interface, plane = self.system.interface_crossed(loc, new_loc)
        crossed = (~np.isnan(plane)) if plane is not None else False
        if np.any(crossed):
            interface_steps = np.where(crossed, (plane - loc[:, 2]) / dir_cos[:, 2], float('inf'))

            # Find photons that should move to the interfaces instead
            move_to_interface = (interface_steps < step) & crossed
            new_loc[move_to_interface] = (loc[move_to_interface] + interface_steps[move_to_interface, np.newaxis]
                                          * dir_cos[move_to_interface])

            # Reflect/refract photons at interfaces
            if np.any(move_to_interface):
                self.reflect_refract(interface, move_to_interface)

        # Update location
        self.location_coordinates = new_loc

        # Update history for all photons
        self.location_history = np.append(self.location_history, new_loc[..., np.newaxis], axis=2)
        self.weight_history = np.append(self.weight_history, self.weight[..., np.newaxis], axis=1)

        # Check if any new photons exited
        exit_mask = (self.headed_into == self.system.surroundings) & (self.weight > 0)

        if np.any(exit_mask):
            self.exit_location[exit_mask] = self.location_history[exit_mask, ..., -1]
            self.exit_weight[exit_mask] = self.weight_history[exit_mask, -1]

            # Check if any exited photons hit a detector
            detector_mask = exit_mask & (self.exit_location[:, 2] == self.system.detector_location)
            if np.any(detector_mask):
                self.system.detector(self, exit_mask)

            # Handle reflection or transmission
            self.R += np.sum(np.where(exit_mask & (self.directional_cosines[:, 2] < 0), self.weight, 0))
            self.T += np.sum(np.where(exit_mask & (self.directional_cosines[:, 2] > 0), self.weight, 0))

            # Terminate exited photons
            self.weight[exit_mask] = 0

    def reflect_refract(self,
                        interfaces: Iterable[Tuple[Medium, Medium]],
                        mask: NDArray[np.bool_]) -> None:
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
        self.A += np.sum(np.where(stop_tir, self.weight[mask], 0))
        self.weight[mask] = np.where(stop_tir, 0, self.weight[mask])

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
        specular_reflection = 0.5 * (rs + rp) * self.weight[mask][refract_mask]

        mu_z_t[refract_mask] = mu_z_t_masked * np.sign(mu_z_i[refract_mask])  # Ensure correct sign

        # Updated for transmitted portion
        mu_x[refract_mask] *= n1_masked / n2_masked
        mu_y[refract_mask] *= n1_masked / n2_masked

        if self.recurse:
            pass
        else:
            # If the reflected fraction will be reflected out, add it to reflected count, Else add it to transmitted
            reflected_out = mu_z_i[refract_mask] > 0
            transmitted_out = mu_z_i[refract_mask] < 0
            self.R += np.sum(specular_reflection * reflected_out)
            self.T += np.sum(specular_reflection * transmitted_out)

            w_temp = self.weight[mask].copy()
            w_temp[refract_mask] -= specular_reflection
            self.weight[mask] = w_temp

        # Send to setter for normalization
        self.directional_cosines[mask] = np.column_stack((mu_x, mu_y, mu_z_t))

    def scatter(self, theta_phi: Optional[Union[Iterable[Real, Real], Iterable[Iterable[Real, Real]]]] = None):
        # Ignore photons at interfaces
        at_interface = np.array([iterable(interface) for interface in self.system.in_medium(self.location_coordinates)])

        # Early break if all are at an interface
        if np.all(at_interface):
            return

        # Placeholders for angle samples
        theta = np.zeros(self.batch_size, dtype=np.float64)
        cosine_theta = np.zeros_like(theta, dtype=np.float64)
        if theta_phi is None:
            # Sample random scattering angles from distribution
            [xi, zeta] = np.random.rand(self.batch_size, 2).T
            g = np.array([medium.g for medium in self.medium])

            # For non-zero g
            non_zero_g_mask = g != 0
            lead_coeff = 1 / (2 * g[non_zero_g_mask])
            term_2 = g[non_zero_g_mask] ** 2
            term_3 = (1 - term_2) / (1 - g[non_zero_g_mask] + (2 * g[non_zero_g_mask] * xi[non_zero_g_mask]))
            cosine_theta[non_zero_g_mask] = lead_coeff * (1 + term_2 - term_3)

            # For g=0
            cosine_theta[~non_zero_g_mask] = (2 * xi[~non_zero_g_mask]) - 1

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
        self.directional_cosines[~at_interface] = new_directional_cosines[~at_interface]

    def plot_path(self, project_onto=None, axes=None, ignore_outside=True):
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


def sample_spectrum(wavelengths: Iterable[Real],
                    spectrum: Iterable[Real]):
    wavelengths = np.asarray(wavelengths)
    spectrum = np.asarray(spectrum)

    # Normalize PDF
    spectrum /= np.sum(spectrum)

    # Compute CDF
    cdf = np.cumsum(spectrum)

    # Take random sample
    i = np.random.uniform(0, 1)

    # Interpolate value of sample from CDF
    return np.interp(i, cdf, wavelengths)
