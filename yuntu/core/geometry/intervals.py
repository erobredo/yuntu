import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Non2DGeometryMixin, \
                                       TimeIntervalMixin, \
                                       FrequencyIntervalMixin

class TimeInterval(
        Non2DGeometryMixin,
        TimeIntervalMixin,
        Geometry):
    name = Geometry.Types.TimeInterval

    def __init__(self, start_time=None, end_time=None, geometry=None):
        if geometry is None:
            geometry = utils.bbox_to_polygon([
                start_time, end_time,
                0, utils.INFINITY])

        super().__init__(geometry=geometry)

        start_time, _, end_time, _ = self.geometry.bounds
        self.end_time = end_time
        self.start_time = start_time

    def to_dict(self):
        data = super().to_dict()
        data['start_time'] = self.start_time
        data['end_time'] = self.end_time
        return data

    @property
    def bounds(self):
        return self.start_time, None, self.end_time, None

    def buffer(self, buffer=None, **kwargs):
        time = utils._parse_args(buffer, 'buffer', 'time', **kwargs)
        start_time = self.start_time - time
        end_time = self.end_time + time
        return TimeInterval(start_time=start_time, end_time=end_time)

    def shift(self, shift=None, **kwargs):
        time = utils._parse_args(shift, 'shift', 'time', **kwargs)
        start_time = self.start_time + time
        end_time = self.end_time + time
        return TimeInterval(start_time=start_time, end_time=end_time)

    def scale(self, scale=None, center=None, **kwargs):
        time = utils._parse_args(scale, 'scale', 'time', **kwargs)

        if center is None:
            center = (self.start_time + self.end_time) / 2
        elif isinstance(center, (list, tuple)):
            center = center[0]

        start_time = center + (self.start_time - center) * time
        end_time = center + (self.end_time - center) * time
        return TimeInterval(start_time=start_time, end_time=end_time)

    def transform(self, transform):
        start_time = transform(self.start_time)
        end_time = transform(self.end_time)
        return TimeInterval(start_time=start_time, end_time=end_time)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        color = kwargs.get('color', None)
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']

        ax.axvline(
            self.start_time,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=color)

        ax.axvline(
            self.end_time,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=color)

        if kwargs.get('fill', True):
            ax.axvspan(
                self.start_time,
                self.end_time,
                alpha=kwargs.get('alpha', 0.5),
                color=color,
                label=kwargs.get('label', None))

        return ax


class FrequencyInterval(
        Non2DGeometryMixin,
        FrequencyIntervalMixin,
        Geometry):
    name = Geometry.Types.FrequencyInterval

    def __init__(self, min_freq=None, max_freq=None, geometry=None):
        if geometry is None:
            geometry = utils.bbox_to_polygon([
                0, utils.INFINITY,
                min_freq, max_freq
            ])

        super().__init__(geometry=geometry)

        _, min_freq, _, max_freq = self.geometry.bounds
        self.min_freq = min_freq
        self.max_freq = max_freq

    def to_dict(self):
        data = super().to_dict()
        data['min_freq'] = self.min_freq
        data['max_freq'] = self.max_freq
        return data

    @property
    def bounds(self):
        return None, self.min_freq, None, self.max_freq

    def buffer(self, buffer=None, **kwargs):
        freq = utils._parse_args(buffer, 'buffer', 'freq', index=1, **kwargs)
        min_freq = self.min_freq - freq
        max_freq = self.max_freq + freq
        return FrequencyInterval(min_freq=min_freq, max_freq=max_freq)

    def shift(self, shift=None, **kwargs):
        freq = utils._parse_args(shift, 'shift', 'freq', index=1, **kwargs)
        min_freq = self.min_freq + freq
        max_freq = self.max_freq + freq
        return FrequencyInterval(min_freq=min_freq, max_freq=max_freq)

    def scale(self, scale=None, center=None, **kwargs):
        freq = utils._parse_args(scale, 'scale', 'freq', index=1, **kwargs)

        if center is None:
            center = (self.min_freq + self.max_freq) / 2
        elif isinstance(center, (list, tuple)):
            center = center[0]

        min_freq = center + (self.min_freq - center) * freq
        max_freq = center + (self.max_freq - center) * freq
        return FrequencyInterval(min_freq=min_freq, max_freq=max_freq)

    def transform(self, transform):
        min_freq = transform(self.min_freq)
        max_freq = transform(self.max_freq)
        return FrequencyInterval(min_freq=min_freq, max_freq=max_freq)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        color = kwargs.get('color', None)
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']

        ax.axhline(
            self.min_freq,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=color)

        ax.axhline(
            self.max_freq,
            linewidth=kwargs.get('linewidth', None),
            linestyle=kwargs.get('linestyle', '--'),
            color=color)

        if kwargs.get('fill', True):
            ax.axhspan(
                self.min_freq,
                self.max_freq,
                alpha=kwargs.get('alpha', 0.5),
                color=color,
                label=kwargs.get('label', None))

        return ax
