import shapely.geometry as shapely_geometry
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Geometry2DMixin, MultiGeometryMixin


class GeometryCollection(
        MultiGeometryMixin,
        Geometry2DMixin,
        Geometry):
    """Point collection geometry."""

    name = Geometry.Types.GeometryCollection

    def __init__(self, collection=None, geometry=None):
        if geometry is None:
            if collection is None:
                message = (
                    'Collection must be provided if no geometry is '
                    'supplied')
                raise ValueError(message)

            geoms = []
            for geom in collection:
                if isinstance(geom, Geometry):
                    geoms.append(geom)
                elif isinstance(geom, (
                        shapely_geometry.Point,
                        shapely_geometry.Polygon,
                        shapely_geometry.LineString
                        )):
                    geoms.append(Geometry.from_geometry(geom))
                elif isinstance(geom, (
                        shapely_geometry.GeometryCollection,
                        shapely_geometry.MultiLineString,
                        shapely_geometry.MultiPoint,
                        shapely_geometry.MultiPolygon
                        )):
                    for sgeom in geom:
                        geoms.append(Geometry.from_geometry(sgeom))
                elif isinstance(geom, GeometryCollection):
                    for sgeom in geom:
                        geoms.append(sgeom)
                else:
                    raise ValueError("All elements of input collection list"
                                     " must be shapely or yuntu geometries.")

            self._geoms = geoms
            geometry = shapely_geometry.GeometryCollection([
                geom.geometry for geom in geoms])

        else:
            self._geoms = [
                Geometry.from_geometry(geom) for geom in geometry.geoms
            ]

        super().__init__(geometry=geometry)

    @property
    def geoms(self):
        for geom in self._geoms:
            yield geom

    def buffer(self, *args, **kwargs):
        return GeometryCollection(collection=[
            geom.buffer(*args, **kwargs)
            for geom in self.geoms
        ])

    def shift(self, *args, **kwargs):
        return GeometryCollection(collection=[
            geom.shift(*args, **kwargs)
            for geom in self.geoms
        ])

    def scale(self, *args, **kwargs):
        return GeometryCollection(collection=[
            geom.scale(*args, **kwargs)
            for geom in self.geoms
        ])

    def transform(
            self,
            transform,
            time_level=0,
            frequency_level=0):
        from yuntu.core.geometry import lines
        from yuntu.core.geometry import intervals

        collection = []
        for geom in self.geoms:
            if isinstance(geom, (
                    lines.TimeLine,
                    intervals.TimeInterval)):
                collection.append(
                    geom.transform(lambda x: transform(x, time_level)[0]))

            elif isinstance(geom, (
                    lines.FrequencyLine,
                    intervals.FrequencyInterval)):
                collection.append(
                    geom.transform(lambda y: transform(frequency_level, y)[1]))

            else:
                collection.append(geom.transform(transform))

        return GeometryCollection(collection)

    def rotate(self, *args, **kwargs):
        return GeometryCollection(collection=[
            geom.rotate(*args, **kwargs)
            for geom in self.geoms
        ])
