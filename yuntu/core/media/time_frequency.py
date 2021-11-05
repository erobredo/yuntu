from typing import Optional

import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

import yuntu.core.windows as windows
from yuntu.core import geometry as geom
from yuntu.core.annotation import annotation
from yuntu.core.media import masked
from yuntu.core.media.base import Media
from yuntu.core.media.time import TimeMediaMixin
from yuntu.core.media.time import TimeItem
from yuntu.core.media.utils import pad_array
from yuntu.core.media.frequency import FrequencyMediaMixin
from yuntu.core.media.frequency import FrequencyItem
import yuntu.core.utils.atlas as geom_utils


class TimeItemWithFrequencies(FrequencyMediaMixin, TimeItem):
    pass


class FrequencyItemWithTime(TimeMediaMixin, FrequencyItem):
    pass


class TimeFrequencyMediaMixin(TimeMediaMixin, FrequencyMediaMixin):
    frequency_axis_index = 0
    time_axis_index = 1

    window_class = windows.TimeFrequencyWindow
    time_item_class = TimeItemWithFrequencies
    frequency_item_class = FrequencyItemWithTime

    plot_xlabel = 'Time (s)'
    plot_ylabel = 'Frequency (Hz)'

    def __init__(
            self,
            start=0,
            duration=None,
            time_resolution=None,
            time_axis=None,
            min_freq=0,
            max_freq=None,
            freq_resolution=None,
            frequency_axis=None,
            window=None,
            **kwargs):

        if time_axis is None:
            time_axis = self.time_axis_class(
                resolution=time_resolution)

        if not isinstance(time_axis, self.time_axis_class):
            time_axis = self.time_axis_class.from_dict(time_axis)

        if frequency_axis is None:
            frequency_axis = self.frequency_axis_class(
                resolution=freq_resolution)

        if not isinstance(frequency_axis, self.frequency_axis_class):
            frequency_axis = self.frequency_axis_class.from_dict(frequency_axis) # noqa

        if window is None:
            window = windows.TimeFrequencyWindow(
                start=start,
                end=duration,
                min=min_freq,
                max=max_freq)

        if not isinstance(window, windows.TimeFrequencyWindow):

            if isinstance(window, windows.TimeWindow):
                window = windows.TimeFrequencyWindow(
                    start=window.start,
                    end=window.end,
                    min=min_freq,
                    max=max_freq)

            elif isinstance(window, windows.FrequencyWindow):
                window = windows.TimeFrequencyWindow(
                    start=start,
                    end=duration,
                    min=window.min,
                    max=window.max)

        super().__init__(
            start=start,
            frequency_axis=frequency_axis,
            time_axis=time_axis,
            window=window,
            **kwargs)

    def get_value(self, time: float, freq: float) -> float:
        """Get media value at a given time and frequency.

        Parameters
        ----------
        time: float
            Time in seconds.
        freq: float
            Frequency in hertz.

        Returns
        -------
        float
            The value of the spectrogram at the desired time and frequency.
        """
        time_index = self.get_index_from_time(time)
        time_index = self._restrain_time_index(time_index)

        freq_index = self.get_index_from_frequency(freq)
        freq_index = self._restrain_freq_index(freq_index)

        if self.time_axis_index > self.frequency_axis_index:
            first_axis = self.time_axis_index
            first_index = time_index

            second_axis = self.frequency_axis_index
            second_index = freq_index
        else:
            first_axis = self.frequency_axis_index
            first_index = freq_index

            second_axis = self.time_axis_index
            second_index = time_index

        result = self.array.take(first_index, axis=first_axis)
        return result.take(second_index, axis=second_axis)

    # pylint: disable=arguments-differ
    def read(
            self,
            start_time=None,
            end_time=None,
            min_freq=None,
            max_freq=None):
        if min_freq is None:
            min_freq = self._get_min()

        if max_freq is None:
            max_freq = self._get_max()

        if min_freq > max_freq:
            message = 'Read min_freq should be less than read max_freq.'
            raise ValueError(message)

        if start_time is None:
            start_time = self._get_start()

        if end_time is None:
            end_time = self._get_end()

        if start_time > end_time:
            message = 'Read start_time should be less than read end_time.'
            raise ValueError(message)

        start_freq_index = self.get_index_from_frequency(min_freq)
        end_freq_index = self.get_index_from_frequency(max_freq)

        start_time_index = self.get_index_from_time(start_time)
        end_time_index = self.get_index_from_time(end_time)

        slices = self._build_slices(
            start_time_index,
            end_time_index + 1,
            start_freq_index,
            end_freq_index + 1)
        return self.array[slices]

    def cut(
            self,
            window: Optional[windows.TimeFrequencyWindow] = None,
            geometry: Optional[geom.Geometry] = None,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
            max_freq: Optional[float] = None,
            min_freq: Optional[float] = None,
            lazy: Optional[bool] = False,
            pad=False,
            pad_mode='constant',
            constant_values=0):
        current_start = self._get_start()
        current_end = self._get_end()
        current_min = self._get_min()
        current_max = self._get_max()

        if window is not None:
            assert isinstance(window, windows.Window)

        if geometry is not None:
            assert isinstance(geometry, geom.Geometry)

        if start_time is None:
            if window is not None and hasattr(window, 'start'):
                start_time = window.start
            elif geometry is not None:
                start_time, _, _, _ = geometry.bounds

            if start_time is None:
                start_time = current_start

        if end_time is None:
            if window is not None and hasattr(window, 'end'):
                end_time = window.end
            elif geometry is not None:
                _, _, end_time, _ = geometry.bounds

            if end_time is None:
                end_time = current_end

        if min_freq is None:
            if window is not None and hasattr(window, 'min'):
                min_freq = window.min
            elif geometry is not None:
                _, min_freq, _, _ = geometry.bounds

            if min_freq is None:
                min_freq = current_min

        if max_freq is None:
            if window is not None and hasattr(window, 'max'):
                max_freq = window.max
            elif geometry is not None:
                _, _, _, max_freq = geometry.bounds

            if max_freq is None:
                max_freq = current_max

        if start_time > end_time or min_freq > max_freq:
            raise ValueError('Cut is empty')

        bounded_start_time = max(min(start_time, current_end), current_start)
        bounded_end_time = max(min(end_time, current_end), current_start)
        bounded_max_freq = max(min(max_freq, current_max), current_min)
        bounded_min_freq = max(min(min_freq, current_max), current_min)

        kwargs = self._copy_dict()
        kwargs['window'] = windows.TimeFrequencyWindow(
            start=start_time if pad else bounded_start_time,
            end=end_time if pad else bounded_end_time,
            min=min_freq if pad else bounded_min_freq,
            max=max_freq if pad else bounded_max_freq)

        if lazy:
            # TODO: No lazy cutting for now. The compute method does not take
            # into acount possible cuts and thus might not give the correct
            # result.
            lazy = False
        kwargs['lazy'] = lazy

        if not lazy:
            kwargs['array'] = self.cut_array(
                window=window,
                start_time=start_time,
                end_time=end_time,
                max_freq=max_freq,
                min_freq=min_freq,
                pad=pad,
                pad_mode=pad_mode,
                constant_values=constant_values)

        return type(self)(**kwargs)

    def cut_array(
            self,
            window: Optional[windows.TimeFrequencyWindow] = None,
            geometry: Optional[geom.Geometry] = None,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
            max_freq: Optional[float] = None,
            min_freq: Optional[float] = None,
            pad=False,
            pad_mode='constant',
            constant_values=0):
        current_start = self._get_start()
        current_end = self._get_end()
        current_min = self._get_min()
        current_max = self._get_max()

        if window is not None:
            assert isinstance(window, windows.Window)

        if geometry is not None:
            assert isinstance(geometry, geom.Geometry)

        if start_time is None:
            if window is not None and hasattr(window, 'start'):
                start_time = window.start
            elif geometry is not None:
                start_time, _, _, _ = geometry.bounds

            if start_time is None:
                start_time = current_start

        if end_time is None:
            if window is not None and hasattr(window, 'end'):
                end_time = window.end
            elif geometry is not None:
                _, _, end_time, _ = geometry.bounds

            if end_time is None:
                end_time = current_end

        if min_freq is None:
            if window is not None and hasattr(window, 'min'):
                min_freq = window.min
            elif geometry is not None:
                _, min_freq, _, _ = geometry.bounds

            if min_freq is None:
                min_freq = current_min

        if max_freq is None:
            if window is not None and hasattr(window, 'max'):
                max_freq = window.max
            elif geometry is not None:
                _, _, _, max_freq = geometry.bounds

            if max_freq is None:
                max_freq = current_max

        if start_time > end_time or min_freq > max_freq:
            raise ValueError('Cut is empty')

        bounded_start_time = max(min(start_time, current_end), current_start)
        bounded_end_time = max(min(end_time, current_end), current_start)
        bounded_max_freq = max(min(max_freq, current_max), current_min)
        bounded_min_freq = max(min(min_freq, current_max), current_min)

        start_index = self.get_index_from_time(bounded_start_time)
        end_index = self.get_index_from_time(bounded_end_time)
        min_index = self.get_index_from_frequency(bounded_min_freq)
        max_index = self.get_index_from_frequency(bounded_max_freq)

        slices = self._build_slices(
                start_index,
                end_index,
                min_index,
                max_index)
        array = self.array[slices]

        if pad:
            start_pad = self.time_axis.get_bin_nums(
                    start_time, bounded_start_time)
            end_pad = self.time_axis.get_bin_nums(
                    bounded_end_time, end_time)

            min_pad = self.frequency_axis.get_bin_nums(
                    min_freq, bounded_min_freq)
            max_pad = self.frequency_axis.get_bin_nums(
                    bounded_max_freq, max_freq)

            pad_widths = self._build_pad_widths(
                    start_pad,
                    end_pad,
                    min_pad,
                    max_pad)

            array = pad_array(
                    array,
                    pad_widths,
                    mode=pad_mode,
                    constant_values=constant_values)

        return array

    def resample(
            self,
            time_resolution=None,
            freq_resolution=None,
            lazy: Optional[bool] = False,
            kind: str = 'linear',
            **kwargs):
        """Get a new FrequencyMedia object with the resampled data."""
        if time_resolution is None:
            time_resolution = self.time_axis.resolution

        if freq_resolution is None:
            freq_resolution = self.frequency_axis.resolution

        data = self._copy_dict()
        data['lazy'] = lazy
        new_time_axis = self.time_axis.resample(time_resolution)
        data['time_axis'] = new_time_axis

        new_freq_axis = self.frequency_axis.resample(freq_resolution)
        data['frequency_axis'] = new_freq_axis

        if not lazy:
            if self.ndim != 2:
                message = (
                    'Media elements with more than 2 dimensions cannot be'
                    ' resampled')
                raise ValueError(message)

            new_times = new_time_axis.get_bins(window=self.window)
            new_freqs = new_freq_axis.get_bins(window=self.window)

            if self.time_axis_index == 1:
                xcoord = self.times
                ycoord = self.frequencies

                newxcoord = new_times
                newycoord = new_freqs
            else:
                xcoord = self.frequencies
                ycoord = self.times

                newxcoord = new_freqs
                newycoord = new_times

            if kind == 'linear':
                interp = interp2d(
                    xcoord,
                    ycoord,
                    self.array,
                    **kwargs)
            else:
                interp = RectBivariateSpline(
                    xcoord,
                    ycoord,
                    self.array,
                    **kwargs)

            data['array'] = interp(newxcoord, newycoord)

        return type(self)(**data)

    def get_freq_item_kwargs(self, freq):
        return {
            'window': self.window.copy(),
            'time_axis': self.time_axis
        }

    def get_time_item_kwargs(self, freq):
        return {
            'window': self.window.copy(),
            'frequency_axis': self.frequency_axis
        }

    def get_aggr_value(
            self,
            time=None,
            freq=None,
            buffer=None,
            bins=None,
            window=None,
            geometry=None,
            aggr_func=np.mean):
        if bins is not None and buffer is not None:
            message = 'Bins and buffer arguments are mutually exclusive.'
            raise ValueError(message)

        if buffer is None and bins is not None:
            if not isinstance(bins, (list, tuple)):
                bins = [bins, bins]

            time_buffer = self.time_axis.resolution * bins[0]
            freq_buffer = self.frequency_axis.resolution * bins[1]
            buffer = [time_buffer, freq_buffer]

        if window is not None:
            if not isinstance(window, windows.Window):
                window = windows.Window.from_dict(window)

            if buffer is not None:
                window = window.buffer(buffer)

            values = self.cut(window=window).array
            return aggr_func(values)

        if time is not None and freq is not None:
            geometry = geom.Point(time, freq)

        if geometry is None:
            message = (
                'Either time and frequency, a window, or a geometry '
                'should be supplied.')
            raise ValueError(message)

        if buffer is not None:
            geometry = geometry.buffer(buffer)

        mask = self.to_mask(geometry)
        return aggr_func(self.array[mask.array])

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
            time_axis=self.time_axis,
            frequency_axis=self.frequency_axis)

    # pylint: disable=arguments-differ
    def _build_slices(self, start_time, end_time, min_freq, max_freq):
        slice_args = [slice(None, None, None) for _ in self.shape]
        slice_args[self.time_axis_index] = slice(start_time, end_time)
        slice_args[self.frequency_axis_index] = slice(min_freq, max_freq)
        return tuple(slice_args)

    def _build_pad_widths(self, start_pad, end_pad, min_pad, max_pad):
        widths = [(0, 0) for _ in self.shape]
        widths[self.time_axis_index] = (start_pad, end_pad)
        widths[self.frequency_axis_index] = (min_pad, max_pad)
        return widths


@masked.masks(TimeFrequencyMediaMixin)
class TimeFrequencyMaskedMedia(TimeFrequencyMediaMixin, masked.MaskedMedia):
    plot_title = 'Time Frequency Masked Object'

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('mask', True):
            ax.pcolormesh(
                self.times,
                self.frequencies,
                self.array,
                cmap=kwargs.get('cmap', 'gray'),
                alpha=kwargs.get('alpha', 1.0))

        return ax

    def compute(self):
        return geom_utils.geometry_to_mask(
            self.geometry.geometry,
            self.media.array.shape,
            transformX=self.media.get_index_from_time,
            transformY=self.media.get_index_from_frequency)


class TimeFrequencyMedia(TimeFrequencyMediaMixin, Media):
    pass
