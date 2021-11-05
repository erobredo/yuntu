import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry
from yuntu.core.geometry.mixins import Geometry2DMixin


class BBox(Geometry2DMixin, Geometry):
    name = Geometry.Types.BBox

    def __init__(
            self,
            start_time=None,
            end_time=None,
            min_freq=None,
            max_freq=None,
            geometry=None):
        if geometry is None:
            if start_time is None:
                message = 'Bounding box start time must be set.'
                raise ValueError(message)

            if end_time is None:
                message = 'Bounding box end time must be set.'
                raise ValueError(message)

            if min_freq is None:
                message = 'Bounding box max freq must be set.'
                raise ValueError(message)

            if max_freq is None:
                message = 'Bounding box min freq must be set.'
                raise ValueError(message)

            geometry = utils.bbox_to_polygon([
                start_time, end_time,
                min_freq, max_freq])

        super().__init__(geometry=geometry)

        start_time, min_freq, end_time, max_freq = self.geometry.bounds
        self.start_time = start_time
        self.end_time = end_time
        self.min_freq = min_freq
        self.max_freq = max_freq

    def to_dict(self):
        data = super().to_dict()
        data['start_time'] = self.start_time
        data['end_time'] = self.end_time
        data['min_freq'] = self.min_freq
        data['max_freq'] = self.max_freq
        return data

    def buffer(self, buffer=None, **kwargs):
        time, freq = utils._parse_tf(buffer, 'buffer', default=0, **kwargs)
        start_time = self.start_time - time
        end_time = self.end_time + time
        min_freq = self.min_freq - freq
        max_freq = self.max_freq + freq
        return BBox(
            start_time=start_time,
            end_time=end_time,
            min_freq=min_freq,
            max_freq=max_freq)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        xcoords, ycoords = self.geometry.exterior.xy
        lineplot, = ax.plot(
            xcoords,
            ycoords,
            linewidth=kwargs.get('linewidth', None),
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '--'),
        )

        color = lineplot.get_color()

        if kwargs.get('fill', True):
            ax.fill(
                xcoords,
                ycoords,
                color=color,
                alpha=kwargs.get('alpha', 0.5),
                linewidth=0,
                label=kwargs.get('label', None))

        return ax
