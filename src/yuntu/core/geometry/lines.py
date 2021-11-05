import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Non2DGeometryMixin, TimeIntervalMixin


class TimeLine(Non2DGeometryMixin, Geometry):
    name = Geometry.Types.TimeLine

    def __init__(self, time=None, geometry=None):
        if geometry is None:
            geometry = utils.linestring_geometry([
                (time, 0),
                (time, utils.INFINITY)])

        super().__init__(geometry=geometry)

        time, _, _, _ = self.geometry.bounds
        self.time = time

    def to_dict(self):
        data = super().to_dict()
        data['time'] = self.time
        return data

    @property
    def bounds(self):
        return self.time, None, self.time, None

    def buffer(self, buffer=None, **kwargs):
        from yuntu.core.geometry.intervals import TimeInterval

        time = utils._parse_args(buffer, 'buffer', 'time', **kwargs)
        start_time = self.time - time
        end_time = self.time + time
        return TimeInterval(
            start_time=start_time,
            end_time=end_time)

    def shift(self, shift=None, **kwargs):
        time = utils._parse_args(shift, 'shift', 'time', **kwargs)
        time = self.time + time
        return TimeLine(time=time)

    def scale(self, scale=None, center=None, **kwargs):
        return self

    def transform(self, transform):
        time = transform(self.time)
        return TimeLine(time=time)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        ax.axvline(
            self.time,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', None),
            label=kwargs.get('label', None))

        return ax


class FrequencyLine(Non2DGeometryMixin, Geometry):
    name = Geometry.Types.FrequencyLine

    def __init__(self, freq=None, geometry=None):
        if geometry is None:
            geometry = utils.linestring_geometry([
                (0, freq),
                (utils.INFINITY, freq)])

        super().__init__(geometry=geometry)

        _, freq, _, _ = self.geometry.bounds
        self.freq = freq

    def to_dict(self):
        data = super().to_dict()
        data['freq'] = self.freq
        return data

    @property
    def bounds(self):
        return None, self.freq, None, self.freq

    def buffer(self, buffer=None, **kwargs):
        from yuntu.core.geometry.intervals import FrequencyInterval

        freq = utils._parse_args(buffer, 'buffer', 'freq', index=1, **kwargs)
        min_freq = self.freq - freq
        max_freq = self.freq + freq
        return FrequencyInterval(
            min_freq=min_freq,
            max_freq=max_freq)

    def shift(self, shift=None, **kwargs):
        freq = utils._parse_args(shift, 'shift', 'freq', index=1, **kwargs)
        freq = self.freq + freq
        return FrequencyLine(freq=freq)

    def scale(self, scale=None, center=None, **kwargs):
        return self

    def transform(self, transform):
        freq = transform(self.freq)
        return FrequencyLine(freq=freq)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        ax.axhline(
            self.freq,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', None),
            label=kwargs.get('label', None))

        return ax
