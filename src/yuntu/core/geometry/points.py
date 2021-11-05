import shapely.geometry as shapely_geometry
import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Geometry2DMixin, MultiGeometryMixin


class Point(Geometry):
    name = Geometry.Types.Point

    def __init__(self, time=None, freq=None, geometry=None):
        if geometry is None:
            geometry = utils.point_geometry(time, freq)

        super().__init__(geometry=geometry)

        self.time = self.geometry.x
        self.freq = self.geometry.y

    def __getitem__(self, key):
        if not isinstance(key, int):
            message = f'Index must be integer not {type(key)}'
            raise ValueError(message)

        if key < 0 or key > 1:
            raise IndexError

        if key == 0:
            return self.time

        return self.freq

    def to_dict(self):
        data = super().to_dict()
        data['time'] = self.time
        data['freq'] = self.freq
        return data

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        ax.scatter(
            [self.time],
            [self.freq],
            s=kwargs.get('size', None),
            color=kwargs.get('color', None),
            marker=kwargs.get('marker', None),
            label=kwargs.get('label', None))

        return ax


class MultiPoint(
        MultiGeometryMixin,
        Geometry2DMixin,
        Geometry):
    """Point collection geometry."""

    name = Geometry.Types.MultiPoint

    def __init__(self, points=None, geometry=None):
        if geometry is None:
            if points is None:
                message = 'Points must be provided if no geometry is supplied'
                raise ValueError(message)

            geoms = []
            for geom in points:
                if isinstance(geom, Point):
                    geoms.append(geom.geometry)
                elif isinstance(geom, shapely_geometry.Point):
                    geoms.append(geom)
                elif isinstance(geom, (list, tuple)):
                    geoms.append(shapely_geometry.Point(*geom))
                else:
                    raise ValueError("All elements of input points list"
                                     " must be points.")

            geometry = shapely_geometry.MultiPoint(geoms)

        super().__init__(geometry=geometry)
