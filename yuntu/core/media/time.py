from typing import Optional
import numpy as np

from yuntu.core.audio.utils import resample
from yuntu.core import geometry as geom
from yuntu.core.annotation import annotation
from yuntu.core import windows
from yuntu.core.media import masked
from yuntu.core.media.utils import pad_array
from yuntu.core.media.base import Media
from yuntu.core.axis import TimeAxis


class TimeItem(Media):
    def __init__(self, time, *args, **kwargs):
        self.freq = time
        super().__init__(*args, **kwargs)

    def load(self, path):
        pass

    def write(self, path):
        pass

    def plot(self, ax=None, **kwargs):
        pass


class TimeMediaMixin:
    time_axis_index = 0
    time_axis_class = TimeAxis
    time_item_class = TimeItem
    window_class = windows.TimeWindow

    plot_xlabel = 'Time (s)'

    def __init__(
            self,
            start=0,
            duration=None,
            resolution=None,
            time_axis=None,
            **kwargs):

        if time_axis is None:
            time_axis = self.time_axis_class(resolution=resolution, **kwargs)

        if not isinstance(time_axis, self.time_axis_class):
            time_axis = self.time_axis_class.from_dict(time_axis)
        self.time_axis = time_axis

        if 'window' not in kwargs:
            kwargs['window'] = windows.TimeWindow(
                start=start,
                end=duration)

        super().__init__(**kwargs)

    @property
    def dt(self):
        return self.time_axis.period

    @property
    def time_size(self):
        if self.is_empty():
            return self.time_axis.get_size(window=self.window)
        return self.array.shape[self.time_axis_index]

    @property
    def times(self):
        """Get the time array.

        This is an array of the same length as the wav data array and holds
        the time (in seconds) corresponding to each piece of the wav array.
        """
        if not self.is_empty():
            size = self.array.shape[self.time_axis_index]
        else:
            size = None
        return self.time_axis.get_bins(window=self.window, size=size)

    def to_dict(self):
        return {
            'time_axis': self.time_axis.to_dict(),
            **super().to_dict()
        }

    def get_index_from_time(self, time):
        """Get the index of the media array corresponding to the given time."""
        start = self._get_start()
        if time < start:
            message = (
                'Time earlier that start of recording file or window start '
                'was requested')
            raise ValueError(message)

        if time > self._get_end():
            message = (
                'Time earlier that start of recording file or window start '
                'was requested')
            raise ValueError(message)

        return self.time_axis.get_index_from_value(
            time,
            window=self.window)


    def _restrain_time_index(self, index):
        size = self.time_size
        return min(max(0, index), size - 1)

    def get_value(self, time):
        index = self.get_index_from_time(time)
        index = self._restrain_time_index(index)
        return self.array.take(index, axis=self.time_axis_index)

    def get_time_item_kwargs(self, freq):
        return {'window': self.window.copy()}

    def get_time_item(self, freq):
        index = self.get_index_from_time(freq)
        index = self._restrain_time_index(index)
        array = self.array.take(index, axis=self.time_axis_index)
        kwargs = self.get_time_item_kwargs(freq)
        return self.time_item_class(freq, array=array, **kwargs)

    def iter_time(self):
        for time in self.times:
            yield self.get_time_item(time)

    def resample(
            self,
            resolution=None,
            samplerate=None,
            lazy: Optional[bool] = False,
            **kwargs):
        """Get a new TemporalMedia object with the resampled data."""
        if samplerate is None and resolution is None:
            message = 'Either resolution or samplerate must be provided'
            raise ValueError(message)

        if resolution is None:
            resolution = samplerate

        data = self._copy_dict()
        data['lazy'] = lazy
        data['time_axis'] = self.time_axis.resample(resolution)

        if not self.path_exists():
            data = resample(
                self.array,
                self.resolution,
                resolution,
                **kwargs)
            data['array'] = data

        return type(self)(**data)

    def read(self, start=None, end=None):
        """Read a section of the media array.

        Parameters
        ----------
        start: float, optional
            Time at which read starts, in seconds. If not provided
            start will be defined as the recording start. Should
            be larger than 0. If a non trivial window is set, the
            provided starting time should be larger that the window
            starting time.
        end: float, optional
            Time at which read ends, in seconds. If not provided
            end will be defined as the recording end. Should be
            less than the duration of the audio. If a non trivial
            window is set, the provided ending time should be
            larger that the window ending time.

        Returns
        -------
        np.array
            The media data contained in the demanded temporal limits.

        Raises
        ------
        ValueError
            When start is less than end, or end is larger than the
            duration of the audio, or start is less than 0. If a non
            trivial window is set, it will also throw an error if
            the requested starting and ending times are smaller or
            larger that those set by the window.
        """
        if start is None:
            start = self._get_start()

        if end is None:
            end = self._get_end()

        if start > end:
            message = 'Read start should be less than read end.'
            raise ValueError(message)

        start_index = self.get_index_from_time(start)
        end_index = self.get_index_from_time(end)
        return self.array[self._build_slices(start_index, end_index + 1)]

    def calculate_mask(self, geometry):
        """Return masked 1d array."""
        start, _, end, _ = geometry.bounds

        start_index = self.get_index_from_time(start)
        end_index = self.get_index_from_time(end)

        mask = np.zeros(self.shape)
        mask[self._build_slices(start_index, end_index + 1)] = 1
        return mask

    def cut(
            self,
            start_time: float = None,
            end_time: float = None,
            window: windows.TimeWindow = None,
            lazy=False,
            pad=False,
            pad_mode='constant',
            constant_values=0):
        """Get a window to the media data.

        Parameters
        ----------
        start: float, optional
            Window starting time in seconds. If not provided
            it will default to the beggining of the recording.
        end: float, optional
            Window ending time in seconds. If not provided
            it will default to the duration of the recording.
        window: TimeWindow, optional
            A window object to use for cutting.
        lazy: bool, optional
            Boolean flag that determines if the fragment loads
            its data lazily.

        Returns
        -------
        Media
            The resulting media object with the correct window set.
        """
        current_start = self._get_start()
        current_end = self._get_end()

        if start_time is None:
            if window is None:
                start_time = current_start
            else:
                start_time = (
                    window.start
                    if window.start is not None
                    else current_start)

        if end_time is None:
            if window is None:
                end_time = current_end
            else:
                end_time = (
                    window.end
                    if window.end is not None
                    else current_end)

        if end_time < start_time:
            message = 'Window is empty'
            raise ValueError(message)

        bounded_start_time = max(min(start_time, current_end), current_start)
        bounded_end_time = max(min(end_time, current_end), current_start)

        kwargs_dict = self._copy_dict()
        kwargs_dict['window'] = windows.TimeWindow(
            start=start_time if pad else bounded_start_time,
            end=end_time if pad else bounded_end_time)

        if lazy:
            # TODO: No lazy cutting for now. The compute method does not take
            # into acount possible cuts and thus might not give the correct
            # result.
            lazy = False
        kwargs_dict['lazy'] = lazy

        if not lazy:
            start = self.get_index_from_time(bounded_start_time)
            end = self.get_index_from_time(bounded_end_time)

            slices = self._build_slices(start, end)
            array = self.array[slices]

            if pad:
                start_pad = self.time_axis.get_bin_nums(
                    start_time, bounded_start_time)
                end_pad = self.time_axis.get_bin_nums(
                    bounded_end_time, end_time)
                pad_widths = self._build_pad_widths(start_pad, end_pad)
                array = pad_array(
                    array,
                    pad_widths,
                    mode=pad_mode,
                    constant_values=constant_values)

            kwargs_dict['array'] = array

        return type(self)(**kwargs_dict)

    def get_aggr_value(
            self,
            time=None,
            buffer=None,
            bins=None,
            window=None,
            geometry=None,
            aggr_func=np.mean):
        if bins is not None and buffer is not None:
            message = 'Bins and buffer arguments are mutually exclusive.'
            raise ValueError(message)

        if buffer is None and bins is not None:
            buffer = self.time_axis.resolution * bins

        if window is None:
            if time is not None:
                geometry = geom.TimeLine(time)

            if geometry is None:
                message = (
                    'Either time, a window, or a geometry '
                    'should be supplied.')
                raise ValueError(message)

            start_time, _, end_time, _ = geometry.bounds
            window = windows.TimeWindow(start=start_time, end=end_time)

        if not isinstance(window, windows.Window):
            window = windows.Window.from_dict(window)

        if buffer is not None:
            window = window.buffer(buffer)

        values = self.cut(window=window).array
        return aggr_func(values)

    def to_mask(self, geometry, lazy=False):
        if isinstance(geometry, (annotation.Annotation, windows.Window)):
            geometry = geometry.geometry

        if not isinstance(geometry, geom.Geometry):
            geometry = geom.Geometry.from_geometry(geometry)

        intersected = geometry.intersection(self.window)

        return self.mask_class(
            media=self,
            geometry=intersected,
            lazy=lazy,
            time_axis=self.time_axis)

    def grid(self, frame_length, hop_length=None, lazy=False):
        from yuntu.core.media.grided import TimeGridedMedia

        return TimeGridedMedia(
            self,
            frame_length=frame_length,
            hop_length=hop_length,
            lazy=lazy,
            window=self.window)

    def _get_start(self):
        return self.time_axis.get_start(window=self.window)

    def _get_end(self):
        return self.time_axis.get_end(window=self.window)

    def _get_axis_info(self):
        return {
            'time_axis': self.time_axis,
            **super()._get_axis_info()
        }

    def _has_trivial_window(self):
        if self.window.start is not None:
            start = self._get_start()

            if start != self.window.start:
                return False

        if self.window.end is not None:
            end = self._get_end()

            if end != self.window.end:
                return False

        return super()._has_trivial_window()

    def _build_slices(self, start, end):
        slices = [slice(None, None) for _ in self.shape]
        slices[self.time_axis_index] = slice(start, end)
        return tuple(slices)

    def _build_pad_widths(self, start, end):
        widths = [(0, 0) for _ in self.shape]
        widths[self.time_axis_index] = (start, end)
        return widths

    def _copy_dict(self):
        return {
            'time_axis': self.time_axis.copy(),
            **super()._copy_dict()
        }


@masked.masks(TimeMediaMixin)
class TimeMaskedMedia(TimeMediaMixin, masked.MaskedMedia):
    def compute(self):
        mask = np.zeros(self.media.shape)
        start_time, _, end_time, _ = self.geometry.bounds
        start_index = self.get_index_from_time(start_time)
        end_index = self.get_index_from_time(end_time)
        slices = self._build_slices(start_index, end_index)
        mask[slices] = 1
        return mask

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('mask', True):
            intervals = self._get_active_intervals()
            for (start, end) in intervals:
                ax.axvline(
                    start,
                    linewidth=kwargs.get('linewidth', 1),
                    linestyle=kwargs.get('linestyle', '--'),
                    color=kwargs.get('color', 'blue'))

                ax.axvline(
                    end,
                    linewidth=kwargs.get('linewidth', 1),
                    linestyle=kwargs.get('linestyle', '--'),
                    color=kwargs.get('color', 'blue'))

                if kwargs.get('fill', True):
                    ax.axvspan(
                        start,
                        end,
                        alpha=kwargs.get('alpha', 0.2),
                        color=kwargs.get('color', 'blue'))

        return ax

    def _get_active_intervals(self):
        init_slices = self._build_slices(0, -1)
        end_slices = self._build_slices(1, None)
        first_slice = self._get_first_slice()

        mask_slice = self.array[first_slice]
        changes = mask_slice[init_slices] * (1 - mask_slice[end_slices])
        change_indices = (np.nonzero(changes)[0] + 1).tolist()

        if mask_slice[0] == 1:
            change_indices.insert(0, 0)

        if mask_slice[-1] == 1:
            change_indices.append(len(mask_slice) - 1)

        assert len(change_indices) % 2 == 0

        times = self.times
        return [
            [times[start], times[end]]
            for start, end in zip(change_indices[::2], change_indices[1::2])
        ]

    def _build_slices(self, start, end):
        slices = [slice(None, None) for _ in self.media.shape]
        slices[self.time_axis_index] = slice(start, end)
        return tuple(slices)

    def _get_first_slice(self):
        slices = [0 for _ in self.media.shape]
        slices[self.time_axis_index] = slice(None, None)
        return tuple(slices)


class TimeMedia(TimeMediaMixin, Media):
    pass
