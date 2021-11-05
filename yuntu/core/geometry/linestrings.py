import numpy as np
import shapely.geometry as shapely_geometry
import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Geometry2DMixin, MultiGeometryMixin



class LineString(Geometry2DMixin, Geometry):
    name = Geometry.Types.LineString

    def __init__(self, wkt=None, vertices=None, geometry=None):
        if geometry is None:
            if wkt is not None:
                geometry = utils.geom_from_wkt(wkt)

            elif vertices is not None:
                geometry = utils.linestring_geometry(vertices)

            else:
                message = (
                    'Either wkt or vertices must be supplied '
                    'to create a LineString geometry.')
                raise ValueError(message)

        super().__init__(geometry=geometry)

        self.wkt = self.geometry.wkt

    def __iter__(self):
        from yuntu.core.geometry.points import Point

        for (x, y) in self.geometry.coords:
            yield Point(x, y)

    def __getitem__(self, key):
        from yuntu.core.geometry.points import Point

        x, y = self.geometry.coords[key]
        return Point(x, y)

    @property
    def start(self):
        return self[0]

    @property
    def end(self):
        return self[-1]

    def to_dict(self):
        data = super().to_dict()
        data['wkt'] = self.wkt
        return data

    def interpolate(self, s, normalized=True, ratio=1):
        from yuntu.core.geometry.points import Point

        geometry = self.geometry
        if ratio != 1:
            geometry = utils.scale_geometry(
                geometry,
                xfact=ratio,
                origin=(0, 0))

        point = geometry.interpolate(s, normalized=normalized)

        if ratio != 1:
            point = utils.scale_geometry(
                point,
                xfact=1/ratio,
                origin=(0, 0))

        return Point(point.x, point.y)

    def resample(self, num_samples, ratio=1):
        geometry = self.geometry

        if ratio != 1:
            geometry = utils.scale_geometry(
                geometry,
                xfact=ratio,
                origin=(0, 0))

        vertices = [
            geometry.interpolate(param, normalized=True)
            for param in np.linspace(0, 1, num_samples, endpoint=True)
        ]

        if ratio != 1:
            vertices = [
                utils.scale_geometry(
                    point,
                    xfact=1/ratio,
                    origin=(0, 0))
                for point in vertices
            ]
        return LineString(vertices=vertices)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        xcoords, ycoords = self.geometry.xy
        lineplot, = ax.plot(
            xcoords,
            ycoords,
            linewidth=kwargs.get('linewidth', None),
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '--'),
            label=kwargs.get('label', None))

        color = lineplot.get_color()

        if kwargs.get('scatter', False):
            ax.scatter(
                xcoords,
                ycoords,
                color=color,
                s=kwargs.get('size', None))

        return ax


class MultiLineString(
        MultiGeometryMixin,
        Geometry2DMixin,
        Geometry):
    """Linestring collection geometry."""

    name = Geometry.Types.MultiLineString

    def __init__(self, linestrings=None, geometry=None):
        if geometry is None:
            if linestrings is None:
                message = (
                    'Linestrings must be provided if no geometry '
                    'is supplied')
                raise ValueError(message)

            geoms = []
            for geom in linestrings:
                if isinstance(geom, LineString):
                    geoms.append(geom.geometry)
                elif isinstance(geom, shapely_geometry.LineString):
                    geoms.append(geom)
                else:
                    raise ValueError("All elements of input linestring "
                                     "list must be linestrings.")

            geometry = shapely_geometry.MultiLineString(geoms)

        super().__init__(geometry)
