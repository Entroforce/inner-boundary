import functools

import astropy.constants as const
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

class FieldLines:
    """
    A collection of :class:`FieldLine`.

    Parameters
    ----------
    field_lines : list of `FieldLine`.
    """
    def __init__(self, field_lines):
        self.field_lines = np.array(field_lines)

    def __getitem__(self, idx):
        return self.field_lines[idx]

    def __len__(self):
        return len(self.field_lines)

    @property
    @functools.lru_cache()
    def polarities(self):
        """
        Magnetic field line polarities. ``0`` for closed, otherwise sign(Br) on
        the solar surface.
        """
        polarities = [fline.polarity for fline in self.field_lines]
        return np.array(polarities, dtype=int)

    @property
    def connectivities(self):
        """
        Field line connectivities. ``1`` for open, ``0`` for closed.
        """
        return np.abs(self.polarities)

    @property
    def expansion_factors(self):
        """
        Expansion factors. Set to NaN for closed field lines.
        """
        return np.array([fline.expansion_factor for fline in self.field_lines])

    @property
    def open_field_lines(self):
        """
        An `OpenFieldLines` object containing open field lines.
        """
        open_idxs = np.where(self.connectivities == 1)[0]
        return OpenFieldLines(np.array(self.field_lines)[open_idxs])

    @property
    def closed_field_lines(self):
        """
        An `ClosedFieldLines` object containing open field lines.
        """
        closed_idxs = np.where(self.connectivities == 0)[0]
        return ClosedFieldLines(self.field_lines[closed_idxs])



class ClosedFieldLines(FieldLines):
    """
    A set of closed field lines.
    """
    def __init__(self, field_lines):
        super().__init__(field_lines)
        if np.any(self.connectivities):
            raise ValueError('Not all field lines are closed')


class FieldLine:
    """
    A single magnetic field line.

    Parameters
    ----------
    x, y, z :
        Field line coordinates in cartesian coordinates.
    output : Output
        The PFSS output through which this field line was traced.
    """
    def __init__(self, x, y, z, output):
        self._x = np.array(x)
        self._y = np.array(y)
        self._z = np.array(z)
        self._r = np.sqrt(self._x**2 + self._y**2 + self._z**2)
        self._output = output
        # Field line is open if one end is on the solar surface and one on
        # the source surface
        atol = 0.1
        if len(self._r) <= 1:
            self._is_open = False
            self._polarity = 0
        else:
            self._is_open = np.abs(self._r[0] - self._r[-1]) > atol
            self._polarity = -np.sign(self._r[0] - self._r[-1]) * self._is_open

    def __len__(self):
        return len(self._x)

    @property
    def coords(self):
        """
        Field line `~astropy.coordinates.SkyCoord`.
        """
        return self._coords(self._x, self._y, self._z, self._output)

    @staticmethod
    def _coords(x, y, z, output):
        r, lat, lon = coord.cartesian_to_spherical(x, y, z)
        r *= const.R_sun
        lon += output._lon0 + 180 * u.deg
        coords = coord.SkyCoord(
            lon, lat, r, frame=output.coordinate_frame)
        return coords

    @property
    def is_open(self):
        """
        Returns ``True`` if one of the field line is connected to the solar
        surface and one to the outer boundary, ``False`` otherwise.
        """
        return self._is_open

    @property
    def polarity(self):
        """
        Magnetic field line polarity.

        Returns
        -------
        pol : int
            0 if the field line is closed, otherwise sign(Br) of the magnetic
            field on the solar surface.
        """
        return self._polarity

    @property
    def solar_footpoint(self):
        """
        Solar surface magnetic field footpoint.

        This is the ends of the magnetic field line that lies on the solar
        surface.

        Returns
        -------
        footpoint : :class:`~astropy.coordinates.SkyCoord`

        Notes
        -----
        For a closed field line, both ends lie on the solar surface. This
        method returns the field line pointing out from the solar surface in
        this case.
        """
        return self.coords[self._solar_coord_index]

    @property
    def source_surface_footpoint(self):
        """
        Solar surface magnetic field footpoint.

        This is the ends of the magnetic field line that lies on the solar
        surface.

        Returns
        -------
        footpoint : :class:`~astropy.coordinates.SkyCoord`

        Notes
        -----
        For a closed field line, both ends lie on the solar surface. This
        method returns the field line pointing out from the solar surface in
        this case.
        """
        return self.coords[self._ss_coord_index]

    @property
    def _ss_coord_index(self):
        """
        Return 0 or -1 depending on which end of the coordinate array is the
        source surface footpoint.
        """
        if self.polarity == 1 or not self.is_open:
            return -1
        else:
            return 0

    @property
    def _solar_coord_index(self):
        return -1 - self._ss_coord_index

