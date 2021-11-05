import librosa
import numpy as np

from yuntu.core.media.base import Media
from yuntu.core.media import time
from yuntu.core.media import frequency
from yuntu.core.media import time_frequency


class GridedMediaMixin:
    grid_axis = -1

    def __init__(self, media, **kwargs):
        self.media = media
        super().__init__(**kwargs)

    @property
    def item_size(self):
        return self.shape[self.grid_axis]


class TimeGridedMediaMixin(
        GridedMediaMixin,
        time.TimeMediaMixin):
    def __init__(
            self,
            media,
            frame_length,
            hop_length=None,
            start_time=None,
            end_time=None,
            resolution=None,
            **kwargs):
        self.frame_length = frame_length

        if hop_length is None:
            hop_length = frame_length / 2
        self.hop_length = hop_length

        if start_time is None:
            start_time = media._get_start()

        if end_time is None:
            end_time = media._get_end()

        if resolution is None:
            duration = end_time - start_time
            length = media.time_size

            frame_index_length = int(frame_length * media.time_axis.resolution)
            hop_index_length = int(hop_length * media.time_axis.resolution)

            n_frames = 1 + (length - frame_index_length) // hop_index_length
            resolution = n_frames / duration

        super().__init__(
            media,
            start=start_time,
            duration=end_time,
            resolution=resolution,
            **kwargs)

    def to_dict(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            **self.to_dict()
        }

    def plot(self, ax=None, media=False, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if media:
            media_kwargs = kwargs.get('media_kwargs', {})
            self.media.plot(ax=ax, **media_kwargs)

    def write(self, path=None):
        pass

    def compute(self):
        ndim = self.media.ndim
        array = self.media.array.copy()

        axis = self.media.time_axis_index
        if (axis != -1) or (axis == ndim - 1):
            array = np.moveaxis(array, axis, -1)

        frame_length = int(self.frame_length * self.media.time_axis.resolution)
        hop_length = int(self.hop_length * self.media.time_axis.resolution)

        frames = librosa.util.frame(
            array,
            frame_length=frame_length,
            hop_length=hop_length)

        if (axis != -1) or (axis == ndim - 1):
            frames = np.moveaxis(frames, axis, -1)

        return frames

    def _copy_dict(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            **self._copy_dict()
        }


class TimeGridedMedia(TimeGridedMediaMixin, Media):
    pass


class FrequencyGridedMediaMixin(
        frequency.FrequencyMediaMixin,
        GridedMediaMixin):
    def __init__(
            self,
            media,
            frame_length,
            hop_length=None,
            min_freq=None,
            max_freq=None,
            resolution=None,
            **kwargs):
        self.frame_length = frame_length

        if hop_length is None:
            hop_length = frame_length / 2
        self.hop_length = hop_length

        if min_freq is None:
            min_freq = media._get_min()

        if max_freq is None:
            max_freq = media._get_max()

        if resolution is None:
            freq_range = max_freq - min_freq
            length = media.time_size

            media_res = media.frequency_axis.resolution
            frame_index_length = int(frame_length * media_res)
            hop_index_length = int(hop_length * media_res)

            n_frames = 1 + (length - frame_index_length) // hop_index_length
            resolution = n_frames / freq_range

        super().__init__(
            media,
            min_freq=min_freq,
            max_freq=max_freq,
            resolution=resolution,
            **kwargs)

    def to_dict(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            **self.to_dict()
        }

    def compute(self):
        ndim = self.media.ndim
        array = self.media.array.copy()

        axis = self.media.frequency_axis_index
        if (axis != -1) or (axis == ndim - 1):
            array = np.moveaxis(array, axis, -1)

        media_res = self.media.frequency_axis.resolution
        frame_length = int(self.frame_length * media_res)
        hop_length = int(self.hop_length * media_res)

        frames = librosa.util.frame(
            array,
            frame_length=frame_length,
            hop_length=hop_length)

        if (axis != -1) or (axis == ndim - 1):
            frames = np.moveaxis(frames, axis, -1)

        return frames

    def _copy_dict(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            **self._copy_dict()
        }


class TimeFrequencyGridedMediaMixin(
        time_frequency.TimeFrequencyMediaMixin,
        GridedMediaMixin):
    def __init__(
            self,
            media,
            time_frame_length,
            freq_frame_length,
            freq_hop_length=None,
            time_hop_length=None,
            **kwargs):
        self.time_frame_length = time_frame_length
        self.freq_frame_length = freq_frame_length

        self.freq_hop_length = freq_hop_length
        self.time_hop_length = time_hop_length

        super().__init__(media, **kwargs)
