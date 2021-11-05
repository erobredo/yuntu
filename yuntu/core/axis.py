import abc

import numpy as np


class Axis(abc.ABC):

    # pylint: disable=unused-argument
    def __init__(
            self,
            resolution,
            **kwargs):
        self.resolution = resolution

    def to_dict(self):
        return {
            'resolution': self.resolution
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @property
    def period(self):
        return 1 / self.resolution

    def get_index_from_value(self, value, start=None, window=None):
        start = self.get_start(start=start, window=window)
        return self.get_bin_nums(start, value)

    @abc.abstractmethod
    def get_start(self, start=None, window=None):
        pass

    @abc.abstractmethod
    def get_end(self, end=None, window=None):
        pass

    def get_size(self, start=None, end=None, window=None):
        start = self.get_start(start=start, window=window)
        end = self.get_end(end=end, window=window)
        return self.get_bin_nums(start, end)

    def get_bin(self, value):
        return int(np.floor(value * self.resolution))

    def get_bin_nums(self, start, end):
        start_bin = self.get_bin(start)
        end_bin = self.get_bin(end)
        return end_bin - start_bin

    def get_bins(self, start=None, end=None, window=None, size=None):
        start = self.get_start(start=start, window=window)
        end = self.get_end(end=end, window=window)

        if size is None:
            size = self.get_bin_nums(start, end)

        return np.linspace(start, end, size)

    def resample(self, resolution):
        data = self.to_dict()
        data['resolution'] = resolution
        return type(self)(**data)

    def copy(self):
        return type(self)(**self.to_dict())


class TimeAxis(Axis):
    def get_start(self, start=None, window=None):
        if window is None:
            assert start is not None
            return start

        if getattr(window, 'start', None) is None:
            assert start is not None
            return start

        return window.start

    def get_end(self, end=None, window=None):
        if window is None:
            assert end is not None
            return end

        if getattr(window, 'end', None) is None:
            assert end is not None
            return end

        return window.end


class FrequencyAxis(Axis):
    def get_start(self, start=None, window=None):
        if window is None:
            assert start is not None
            return start

        if getattr(window, 'min', None) is None:
            assert start is not None
            return start

        return window.min

    def get_end(self, end=None, window=None):
        if window is None:
            assert end is not None
            return end

        if getattr(window, 'max', None) is None:
            assert end is not None
            return end

        return window.max
