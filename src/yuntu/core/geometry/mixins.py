class Non2DGeometryMixin:
    def union(self, other):
        import yuntu.core.geometry.geometry_collections as geometry_collections

        return geometry_collections.GeometryCollection([self, other])

    def rotate(self, *args, **kwargs):
        message = f'Geometry of type {type(self)} cannot rotate.'
        raise ValueError(message)


class TimeIntervalMixin:
    def to_start_line(self):
        from yuntu.core.geometry.lines import TimeLine

        start_time, _, _, _ = self.bounds
        return TimeLine(time=start_time)

    def to_end_line(self):
        from yuntu.core.geometry.lines import TimeLine

        _, _, end_time, _ = self.bounds
        return TimeLine(time=end_time)

    def to_time_center_line(self):
        from yuntu.core.geometry.lines import TimeLine

        start_time, _, end_time, _ = self.bounds
        center = (start_time + end_time) / 2
        return TimeLine(time=center)


class FrequencyIntervalMixin:
    def to_min_line(self):
        from yuntu.core.geometry.lines import FrequencyLine

        _, min_freq, _, _ = self.bounds
        return FrequencyLine(freq=min_freq)

    def to_max_line(self):
        from yuntu.core.geometry.lines import FrequencyLine

        _, _, _, max_freq = self.bounds
        return FrequencyLine(freq=max_freq)

    def to_freq_center_line(self):
        from yuntu.core.geometry.lines import FrequencyLine

        _, min_freq, _, max_freq = self.bounds
        center = (min_freq + max_freq) / 2
        return FrequencyLine(freq=center)


class Geometry2DMixin(TimeIntervalMixin, FrequencyIntervalMixin):
    def to_time_interval(self):
        from yuntu.core.geometry.intervals import TimeInterval

        start_time, _, end_time, _ = self.bounds
        return TimeInterval(
            start_time=start_time,
            end_time=end_time)

    def to_freq_interval(self):
        from yuntu.core.geometry.intervals import FrequencyInterval

        _, min_freq, _, max_freq = self.bounds
        return FrequencyInterval(
            min_freq=min_freq,
            max_freq=max_freq)

    def to_center(self):
        from yuntu.core.geometry.points import Point

        start_time, min_freq, end_time, max_freq = self.bounds
        time = (start_time + end_time) / 2
        freq = (min_freq + max_freq) / 2
        return Point(time=time, freq=freq)

    def to_bbox(self):
        from yuntu.core.geometry.bbox import BBox

        start_time, min_freq, end_time, max_freq = self.bounds
        return BBox(
            start_time=start_time,
            min_freq=min_freq,
            end_time=end_time,
            max_freq=max_freq)


class MultiGeometryMixin:
    """Mixin that adds multi geometry behaviour to geometry classes."""

    @property
    def geoms(self):
        """Return iterator of geometries."""
        from yuntu.core.geometry.base import Geometry

        for geom in self.geometry.geoms:
            yield Geometry.from_geometry(geom)

    def __iter__(self):
        for geom in self.geoms:
            yield geom

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)
        for geom in self.geoms:
            ax = geom.plot(ax, **kwargs)
        return ax
