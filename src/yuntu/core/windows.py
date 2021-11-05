"""Windows module."""
from typing import Optional
from abc import ABC
from abc import abstractmethod
import yuntu.core.utils.atlas as geom_utils


INFINITY = 10e+15


class Window(ABC):
    """A window is an object used to select portions of data."""
    def __init__(self, geometry=None):
        self.geometry = geometry

    def cut(self, other):
        """Use window to cut out object."""
        return other.cut(window=self)

    @abstractmethod
    def to_dict(self):
        """Return a dictionary representation of the window."""

    @abstractmethod
    def buffer(self, buffer):
        """Get a buffer window."""

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        """Get a buffer window."""

    def copy(self):
        """Copy window object."""
        return self.from_dict(self.to_dict())

    def to_dict(self):
        return {
            'type': type(self).__name__
        }

    def is_trivial(self):
        return True

    @classmethod
    def from_dict(cls, data):
        """Rebuild the window from dictionary data."""
        if 'type' not in data:
            raise ValueError('Window data does not have a type.')

        window_type = data.pop('type')
        if window_type == 'TimeWindow':
            return TimeWindow(**data)

        if window_type == 'FrequencyWindow':
            return FrequencyWindow(**data)

        if window_type == 'TimeFrequencyWindow':
            return TimeFrequencyWindow(**data)

        message = (
            f'Window type {window_type} is incorrect. Valid options: '
            'TimeWindow, FrequencyWindow, TimeFrequencyWindow')
        raise ValueError(message)

    def __repr__(self):
        data = self.to_dict()
        window_type = data.pop('type')
        args = ', '.join([f'{key}={value}' for key, value in data.items()])
        return f'{window_type}({args})'


class TimeWindow(Window):
    """Time window class.

    Used to cut a time interval.
    """

    def __init__(
            self,
            start: Optional[float] = None,
            end: Optional[float] = None,
            **kwargs):
        """Construct a time window.

        Parameters
        ----------
        start: float
            Interval starting time in seconds.
        end:
            Interval ending time in seconds.
        """
        self.start = start
        self.end = end

        if 'geometry' not in kwargs:
            if start is None:
                start = 0

            if end is None:
                end = INFINITY

            kwargs['geometry'] = geom_utils.bbox_to_polygon([
                start, end,
                0, INFINITY
            ])

        super().__init__(**kwargs)

    def plot(self, ax=None, **kwargs):
        """Plot time window."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (15, 5)))

        ax.axvline(
            self.start,
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'blue'))

        ax.axvline(
            self.end,
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'blue'))

        if kwargs.get('fill', True):
            ax.axvspan(
                self.start,
                self.end,
                alpha=kwargs.get('alpha', 0.2),
                color=kwargs.get('color', 'blue'))

        return ax

    def buffer(self, buffer):
        """Get a buffer window."""
        if isinstance(buffer, (tuple, list)):
            buffer = buffer[0]

        start = self.start - buffer
        end = self.end + buffer
        return TimeWindow(start=start, end=end)

    def to_dict(self):
        """Get dictionary representation of window."""
        return {
            'start': self.start,
            'end': self.end,
            **super().to_dict()
        }

    def is_trivial(self):
        """Return if window is trivial."""
        if self.start is not None:
            return False

        if self.end is not None:
            return False

        return super().is_trivial()


class FrequencyWindow(Window):
    """Frequency window class.

    Used to cut a range of frequencies.
    """

    # pylint: disable=redefined-builtin
    def __init__(
            self,
            min: Optional[float] = None,
            max: Optional[float] = None,
            **kwargs):
        """Construct a frequency window.

        Parameters
        ----------
        min: float
            Interval starting frequency in hertz.
        max:
            Interval ending frequency in hertz.
        """
        self.min = min
        self.max = max

        if 'geometry' not in kwargs:
            if min is None:
                min = 0

            if max is None:
                max = INFINITY

            kwargs['geometry'] = geom_utils.bbox_to_polygon([
                0, INFINITY,
                min, max
            ])

        super().__init__(**kwargs)

    def buffer(self, buffer):
        """Get a buffer window."""
        if isinstance(buffer, (tuple, list)):
            buffer = buffer[1]

        min = self.min - buffer
        max = self.max + buffer
        return FrequencyWindow(min=min, max=max)

    def plot(self, ax=None, **kwargs):
        """Plot frequency window."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (15, 5)))

        ax.axhline(
            self.min,
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'blue'))

        ax.axhline(
            self.max,
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'blue'))

        if kwargs.get('fill', True):
            ax.axhspan(
                self.min,
                self.max,
                alpha=kwargs.get('alpha', 0.2),
                color=kwargs.get('color', 'blue'))

        return ax

    def to_dict(self):
        """Get dictionary representation of window."""
        return {
            'min': self.min,
            'max': self.max,
            **super().to_dict()
        }

    def is_trivial(self):
        """Return if window is trivial."""
        if self.min is not None:
            return False

        if self.max is not None:
            return False

        return super().is_trivial()


class TimeFrequencyWindow(TimeWindow, FrequencyWindow):
    """Time and Frequency window class.

    Used to cut a range of frequencies and times.
    """

    # pylint: disable=redefined-builtin
    def __init__(
            self,
            start: Optional[float] = None,
            end: Optional[float] = None,
            min: Optional[float] = None,
            max: Optional[float] = None,
            **kwargs):
        """Construct a time frequency window.

        Parameters
        ----------
        start: float
            Interval starting time in seconds.
        end:
            Interval ending time in seconds.
        min: float
            Interval starting frequency in hertz.
        max:
            Interval ending frequency in hertz.
        """
        if 'geometry' not in kwargs:
            start_time = start if start is not None else 0
            end_time = end if end is not None else INFINITY
            min_freq = min if min is not None else 0
            max_freq = max if max is not None else INFINITY
            kwargs['geometry'] = geom_utils.bbox_to_polygon([
                start_time, end_time,
                min_freq, max_freq
            ])

        super().__init__(start=start, end=end, min=min, max=max, **kwargs)

    def to_time(self):
        return TimeWindow(start=self.start, end=self.end)

    def to_frequency(self):
        return FrequencyWindow(min=self.min, max=self.max)

    def buffer(self, buffer):
        """Get a buffer window."""
        if isinstance(buffer, (int, float)):
            buffer = [buffer, buffer]

        min = self.min - buffer[1]
        max = self.max + buffer[1]
        start = self.start - buffer[0]
        end = self.end + buffer[0]
        return TimeFrequencyWindow(min=min, max=max, start=start, end=end)

    def plot(self, ax=None, **kwargs):
        """Plot frequency window."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (15, 5)))

        xcoords = [self.start, self.end, self.end, self.start, self.start]
        ycoords = [self.min, self.min, self.max, self.max, self.min]

        ax.plot(
            xcoords,
            ycoords,
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'blue'))

        if kwargs.get('fill', True):
            ax.fill(
                xcoords,
                ycoords,
                linewidth=0,
                alpha=kwargs.get('alpha', 0.2),
                color=kwargs.get('color', 'blue'))

        return ax
