from typing import Optional
import numpy as np

import yuntu.core.windows as windows
from yuntu.core import geometry as geom
from yuntu.core.annotation import annotation
from yuntu.core.media import masked
from yuntu.core.audio.utils import resample
from yuntu.core.media.utils import pad_array
from yuntu.core.media.base import Media
from yuntu.core.axis import FrequencyAxis


class FrequencyItem(Media):
    def __init__(self, freq, *args, **kwargs):
        self.freq = freq
        super().__init__(*args, **kwargs)

    def load(self, path):
        pass

    def write(self, path):
        pass

    def plot(self, ax=None, **kwargs):
        pass


class FrequencyMediaMixin:
    frequency_axis_index = 0
    frequency_axis_class = FrequencyAxis
    frequency_item_class = FrequencyItem
    window_class = windows.FrequencyWindow

    plot_ylabel = 'Frequency (Hz)'

    def __init__(
            self,
            min_freq=0,
            max_freq=None,
            resolution=None,
            frequency_axis=None,
            window=None,
            **kwargs):

        if frequency_axis is None:
            frequency_axis = self.frequency_axis_class(
                resolution=resolution,
                **kwargs)

        if not isinstance(frequency_axis, self.frequency_axis_class):
            frequency_axis = self.frequency_axis_class.from_dict(frequency_axis)  # noqa

        self.frequency_axis = frequency_axis

        if window is None:
            window = windows.FrequencyWindow(
                min=min_freq,
                max=max_freq)

        if not isinstance(window, windows.FrequencyWindow):
            window = windows.FrequencyWindow(
                min=min_freq,
                max=max_freq)

        super().__init__(window=window, **kwargs)

    def to_dict(self):
        return {
            'frequency_axis': self.frequency_axis.to_dict(),
            **super().to_dict()
        }

    @property
    def df(self):
        return 1 / self.frequency_axis.resolution

    @property
    def frequency_size(self):
        if self.is_empty():
            return self.frequency_axis.get_size(window=self.window)
        return self.array.shape[self.frequency_axis_index]

    @property
    def resolution(self):
        return self.frequency_axis.resolution

    def get_index_from_frequency(self, freq):
        """Get index of the media array corresponding to a given frequency."""
        minimum = self._get_min()
        if freq < minimum:
            message = (
                'Frequency less than minimum or window minimum '
                'was requested')
            raise ValueError(message)

        if freq > self._get_max():
            message = (
                'Frequency greater than maximum frequency or window maximum '
                'was requested')
            raise ValueError(message)

        return self.frequency_axis.get_index_from_value(
            freq,
            window=self.window)

    def _restrain_freq_index(self, index):
        size = self.frequency_size
        return min(max(0, index), size - 1)

    def get_value(self, freq):
        index = self.get_index_from_frequency(freq)
        index = self._restrain_freq_index(index)
        return self.array.take(index, axis=self.frequency_axis_index)

    def get_freq_item_kwargs(self, freq):
        return {'window': self.window.copy()}

    def get_freq_item(self, freq):
        index = self.get_index_from_frequency(freq)
        index = self._restrain_freq_index(index)
        array = self.array.take(index, axis=self.frequency_axis_index)
        kwargs = self.get_freq_item_kwargs(freq)
        return self.frequency_item_class(freq, array=array, **kwargs)

    def iter_freq(self):
        for freq in self.frequencies:
            yield self.get_freq_item(freq)

    @property
    def frequencies(self):
        """Get the frequency array.

        This is an array of the same length as the media data array and holds
        the frequency (in hertz) corresponding to each piece of the media
        array.
        """
        if not self.is_empty():
            size = self.array.shape[self.frequency_axis_index]
        else:
            size = None
        return self.frequency_axis.get_bins(window=self.window, size=size)

    def resample(
            self,
            resolution: int,
            lazy: Optional[bool] = False,
            **kwargs):
        """Get a new FrequencyMedia object with the resampled data."""
        data_kwargs = self._copy_dict()
        data_kwargs['lazy'] = lazy
        data_kwargs['frequency_axis'] = self.frequency_axis.resample(resolution)

        if not self.path_exists():
            data = resample(
                self.array,
                self.frequency_axis.resolution,
                resolution,
                **kwargs)
            data_kwargs['array'] = data

        return type(self)(**data_kwargs)

    def read(self, min_freq=None, max_freq=None):
        """Read a section of the media array.

        Parameters
        ----------
        min_freq: float, optional
            Frequency at which read starts, in hertz. If not provided
            start will be defined as the minimum frequency. Should
            be larger than 0. If a non trivial window is set, the
            provided starting time should be larger that the window
            starting time.
        max_freq: float, optional
            Frequency at which read ends, in hertz. If not provided
            end will be defined as the maximum frequency. Should be
            less than the duration of the audio. If a non trivial
            window is set, the provided ending time should be
            larger that the window ending time.

        Returns
        -------
        np.array
            The media data contained in the demanded frequency limits.

        Raises
        ------
        ValueError
            When min_freq is less than minimum frequency, or max_freq is
            larger than the maximum frequency stored, or min_freq is less
            than 0. If a non trivial window is set, it will also throw an
            error if the requested minimum and maximum frequencies are smaller
            or larger that those set by the window.
        """
        if min_freq is None:
            min_freq = self._get_min()

        if max_freq is None:
            max_freq = self._get_max()

        if min_freq > max_freq:
            message = 'Read min_freq should be less than read max_freq.'
            raise ValueError(message)

        start_index = self.get_index_from_frequency(min_freq)
        end_index = self.get_index_from_frequency(max_freq)

        return self.array[self._build_slices(start_index, end_index + 1)]

    def cut(
            self,
            min_freq: float = None,
            max_freq: float = None,
            window: windows.TimeWindow = None,
            lazy=False,
            pad=False,
            pad_mode='constant',
            constant_values=0):
        """Get a window to the media data.

        Parameters
        ----------
        min_freq: float, optional
            Window minimum frequency in Hertz. If not provided
            it will default to the minimum frequency.
        max_freq: float, optional
            Window maximum frequency in Hertz. If not provided
            it will default to the maximum frequency
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
        current_min = self._get_min()
        current_max = self._get_max()

        if min_freq is None:
            min_freq = (
                window.min
                if window.min is not None
                else min_freq)

        if max_freq is None:
            max_freq = (
                window.max
                if window.max is not None
                else max_freq)

        if max_freq < min_freq:
            message = 'Window is empty'
            raise ValueError(message)

        bounded_min_freq = max(min(min_freq, current_max), current_min)
        bounded_max_freq = max(min(max_freq, current_max), current_min)

        kwargs_dict = self._copy_dict()
        kwargs_dict['window'] = windows.FrequencyWindow(
            min=min_freq if pad else bounded_min_freq,
            max=max_freq if pad else bounded_max_freq)

        if lazy:
            # TODO: No lazy cutting for now. The compute method does not take
            # into acount possible cuts and thus might not give the correct
            # result.
            lazy = False
        kwargs_dict['lazy'] = lazy

        if not lazy:
            start = self.get_index_from_frequency(bounded_min_freq)
            end = self.get_index_from_frequency(bounded_max_freq)

            slices = self._build_slices(start, end)
            array = self.array[slices]

            if pad:
                min_pad = self.frequency_axis.get_bin_nums(
                    min_freq, bounded_min_freq)
                max_pad = self.frequency_axis.get_bin_nums(
                    bounded_max_freq, max_freq)
                pad_widths = self._build_pad_widths(min_pad, max_pad)
                array = pad_array(
                    array,
                    pad_widths,
                    mode=pad_mode,
                    constant_values=constant_values)

            kwargs_dict['array'] = array

        return type(self)(**kwargs_dict)

    def get_aggr_value(
            self,
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
            buffer = self.frequency_axis.resolution * bins

        if window is None:
            if freq is not None:
                geometry = geom.FrequencyLine(freq)

            if geometry is None:
                message = (
                    'Either freq, a window, or a geometry '
                    'should be supplied.')
                raise ValueError(message)

            _, min_freq, _, max_freq = geometry.bounds
            window = windows.FrequencyWindow(min=min_freq, max=max_freq)

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
            frequency_axis=self.frequency_axis)

    def _build_slices(self, start, end):
        slices = [slice(None, None) for _ in self.shape]
        slices[self.frequency_axis_index] = slice(start, end)
        return tuple(slices)

    def _build_pad_widths(self, start, end):
        widths = [(0, 0) for _ in self.shape]
        widths[self.frequency_axis_index] = (start, end)
        return widths

    def _copy_dict(self):
        return {
            'frequency_axis': self.frequency_axis.copy(),
            **super()._copy_dict()
        }

    def _get_min(self):
        return self.frequency_axis.get_start(window=self.window)

    def _get_max(self):
        return self.frequency_axis.get_end(window=self.window)

    def _get_axis_info(self):
        return {
            'frequency_axis': self.frequency_axis.copy(),
            **super()._get_axis_info()
        }

    def _has_trivial_window(self):
        if self.window.min is not None:
            min_freq = self._get_min()

            if min_freq != self.window.min:
                return False

        if self.window.max is not None:
            max_freq = self._get_max()

            if max_freq != self.window.max:
                return False

        return super()._has_trivial_window()


@masked.masks(FrequencyMediaMixin)
class FrequencyMaskedMedia(FrequencyMediaMixin, masked.MaskedMedia):
    def compute(self, path=None):
        mask = np.zeros(self.media.shape)
        _, min_freq, _, max_freq = self.geometry.bounds
        start_index = self.get_index_from_frequency(min_freq)
        end_index = self.get_index_from_frequency(max_freq)
        slices = self._build_slices(start_index, end_index)
        mask[slices] = 1
        return mask

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('mask', True):
            intervals = self._get_active_intervals()
            for (start, end) in intervals:
                ax.axhline(
                    start,
                    linewidth=kwargs.get('linewidth', 1),
                    linestyle=kwargs.get('linestyle', '--'),
                    color=kwargs.get('color', 'blue'))

                ax.axhline(
                    end,
                    linewidth=kwargs.get('linewidth', 1),
                    linestyle=kwargs.get('linestyle', '--'),
                    color=kwargs.get('color', 'blue'))

                if kwargs.get('fill', True):
                    ax.axhspan(
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

        frequencies = self.frequencies
        return [
            [frequencies[start], frequencies[end]]
            for start, end in zip(change_indices[::2], change_indices[1::2])
        ]

    def _build_slices(self, start, end):
        slices = [slice(None, None) for _ in self.media.shape]
        slices[self.frequency_axis_index] = slice(start, end)
        return tuple(slices)

    def _get_first_slice(self):
        slices = [0 for _ in self.media.shape]
        slices[self.frequency_axis_index] = slice(None, None)
        return tuple(slices)


class FrequencyMedia(FrequencyMediaMixin, Media):
    pass
