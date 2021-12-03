"""Base classes for audio manipulation."""
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union
from uuid import uuid4
from collections import namedtuple
from collections import OrderedDict
import os

import numpy as np

from yuntu.core.media.time import TimeMedia
from yuntu.core.audio.utils import read_info
from yuntu.core.audio.utils import read_media
from yuntu.core.audio.utils import write_media
import yuntu.core.audio.audio_features as features
from yuntu.core.windows import TimeWindow
from yuntu.core.geometry import TimeInterval

CHANNELS = 'nchannels'
SAMPLE_WIDTH = 'sampwidth'
SAMPLE_RATE = 'samplerate'
LENGTH = 'length'
FILE_SIZE = 'filesize'
DURATION = 'duration'
MEDIA_INFO_FIELDS = [
    CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_RATE,
    LENGTH,
    FILE_SIZE,
    DURATION,
]
REQUIRED_MEDIA_INFO_FIELDS = [
    DURATION,
    SAMPLE_RATE
]
OUT_RANGE = {
    "minmax": lambda x1,x2,y1,y2: [min(x1,y1), max(x2,y2)],
    "left": lambda x1,x2,y1,y2: [x1,x2],
    "right": lambda x1,x2,y1,y2: [y1,y2]
}

MediaInfo = namedtuple('MediaInfo', MEDIA_INFO_FIELDS)
MediaInfoType = Dict[str, Union[int, float]]


def media_info_is_complete(media_info: MediaInfoType) -> bool:
    """Check if media info has all required fields."""
    for field in REQUIRED_MEDIA_INFO_FIELDS:
        if field not in media_info:
            return False

    return True


class Audio(TimeMedia):
    """Base class for all audio."""

    features_class = features.AudioFeatures

    # pylint: disable=redefined-builtin, invalid-name
    def __init__(
            self,
            path: Optional[str] = None,
            array: Optional[np.array] = None,
            timeexp: Optional[int] = 1,
            media_info: Optional[MediaInfoType] = None,
            metadata: Optional[Dict[str, Any]] = None,
            id: Optional[str] = None,
            samplerate: Optional[int] = None,
            duration: Optional[float] = None,
            resolution: Optional[float] = None,
            **kwargs):
        """Construct an Audio object.

        Parameters
        ----------
        path: str, optional
            Path to audio file.
        array: np.array, optional
            Numpy array with audio data
        timeexp: int, optional
            Time expansion factor of audio file. Will default
            to 1.
        media_info: dict, optional
            Dictionary holding all audio file media information.
            This information consists of number of channels (nchannels),
            sample width in bytes (sampwidth), sample rate in Hertz
            (samplerate), length of wav array (length), duration of
            audio in seconds (duration), and file size in bytes (filesize).
        metadata: dict, optional
            A dictionary holding any additional information on the
            audio file.
        id: str, optional
            A identifier for this audiofile.
        lazy: bool, optional
            A boolean flag indicating whether loading of audio data
            is done only when required. Defaults to false.
        samplerate: int, optional
            The samplerate used to read the audio data. If different
            from the native sample rate, the audio will be resampled
            at read.
        """
        if path is None and array is None:
            message = 'Either array or path must be supplied'
            raise ValueError(message)

        if timeexp is None:
            timeexp = 1.0

        self.path = path
        self._timeexp = timeexp

        if id is None:
            if path is not None:
                id = os.path.basename(path)
            else:
                id = "array"
        self.id = id

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        if samplerate is None:
            samplerate = resolution

        if ((samplerate is None) or (duration is None)) and media_info is None and self.path is not None:
            media_info = self.read_info()

        if media_info is None and array is not None and self.path is None:
            if samplerate is None and duration is None and array is not None:
                raise ValueError("When creating audio from numpy arrays," +
                                 "either samplerate or duration must be specified.")

            if samplerate is not None and duration is None:
                duration = len(array)/samplerate

            if samplerate is None and duration is not None:
                samplerate = np.round(float(len(array))/duration).astype(int)

        if media_info is not None and isinstance(media_info, dict):
            if not media_info_is_complete(media_info):
                message = (
                    f'Media info is not complete. Provided media info'
                    f'{media_info}. Required fields: {str(MEDIA_INFO_FIELDS)}')
                raise ValueError(message)
            media_info = MediaInfo(**media_info)

        self.media_info = media_info

        if samplerate is None:
            samplerate = self.media_info.samplerate

        if duration is None:
            duration = self.media_info.duration

        if resolution is None:
            resolution = samplerate

        self.features = self.features_class(self)

        super().__init__(
            array=array,
            path=self.path,
            duration=duration,
            resolution=resolution,
            **kwargs)

    @property
    def timeexp(self):
        return self._timeexp

    @timeexp.setter
    def timeexp(self, value):
        if self.is_empty():
            self.force_load()

        prev_timeexp = self._timeexp
        ratio = prev_timeexp / value
        self.time_axis.resolution *= ratio
        self.window.start /= ratio
        self.window.end /= ratio
        self._timeexp = value

    @property
    def samplerate(self):
        return self.time_axis.resolution

    @property
    def duration(self):
        return self.window.end - self.window.start

    @classmethod
    def from_instance(
            cls,
            recording,
            lazy: Optional[bool] = False,
            samplerate: Optional[int] = None,
            **kwargs):
        """Create a new Audio object from a database recording instance."""
        data = {
            'path': recording.path,
            'timeexp': recording.timeexp,
            'media_info': recording.media_info,
            'metadata': recording.metadata,
            'lazy': lazy,
            'samplerate': samplerate,
            'id': recording.id,
            **kwargs
        }
        return cls(**data)

    @classmethod
    def from_dict(
            cls,
            dictionary: Dict[Any, Any],
            lazy: Optional[bool] = False,
            samplerate: Optional[int] = None,
            **kwargs):
        """Create a new Audio object from a dictionary of metadata."""
        if 'path' not in dictionary:
            message = 'No path was provided in the dictionary argument'
            raise ValueError(message)

        dictionary['lazy'] = lazy

        if samplerate is not None:
            dictionary['samplerate'] = samplerate

        return cls(**dictionary, **kwargs)

    @classmethod
    def from_array(
            cls,
            array: np.array,
            samplerate: int,
            metadata: Optional[dict] = None,
            **kwargs):
        """Create a new Audio object from a numpy array."""
        shape = array.shape
        if len(shape) == 1:
            channels = 1
            size = len(array)
        elif len(shape) == 2:
            channels = shape[0]
            size = shape[1]
        else:
            message = (
                f'The array has {len(shape)} dimensions. Could not be '
                'interpreted as an audio array')
            raise ValueError(message)

        media_info = {
            SAMPLE_RATE: samplerate,
            SAMPLE_WIDTH: 16,
            CHANNELS: channels,
            LENGTH: size,
            FILE_SIZE: size * 16 * channels,
            DURATION: size / samplerate
        }

        return cls(
            array=array,
            media_info=media_info,
            id=str(uuid4()),
            metadata=metadata,
            **kwargs)

    def _copy_dict(self, **kwargs):
        return {
            'timeexp': self.timeexp,
            'media_info': self.media_info,
            'metadata': self.metadata,
            'id': self.id,
            **super()._copy_dict(**kwargs),
        }

    def read_info(self, path=None):
        if path is None:
            if self.is_remote():
                path = self.remote_load()
                self._buffer = path
                return read_info(path, self.timeexp)

            path = self.path
        return read_info(path, self.timeexp)

    def load_from_path(self, path=None):
        """Read signal from file."""
        if path is None:
            path = self.path

        if hasattr(self, '_buffer'):
            path = self._buffer

        start = self._get_start()
        end = self._get_end()
        duration = end - start

        signal, _ = read_media(
            path,
            self.samplerate,
            offset=start,
            duration=duration)

        if hasattr(self, '_buffer'):
            self._buffer.close()
            del self._buffer

        return signal

    # pylint: disable=arguments-differ
    def write(
            self,
            path: str,
            media_format: Optional[str] = "wav",
            samplerate: Optional[int] = None):
        """Write media to path."""
        self.path = path

        signal = self.array
        if samplerate is None:
            samplerate = self.samplerate

        write_media(self.path,
                    signal,
                    samplerate,
                    self.media_info.nchannels,
                    media_format)

    def listen(self, speed: Optional[float] = 1):
        """Return HTML5 audio element player of current audio."""
        # pylint: disable=import-outside-toplevel
        from IPython.display import Audio as HTMLAudio
        rate = self.samplerate * speed
        return HTMLAudio(data=self.array, rate=rate)

    def blend(self, other, weight_range=None, weights=None, normalize_first=False, new_range='left'):
        '''Blend this audio signal with another signal and produce Audio object'''
        if self.samplerate != other.samplerate:
            raise ValueError("Both pieces should have the same samplerate")

        if weights is None and weight_range is None:
            weights = [0.5, 0.5]
        elif weight_range is not None:
            l = np.random.uniform(weight_range[0], weight_range[1], 1)[0]
            weights = [l, 1-l]
        elif weights is not None:
            if weights[0] > weights[1]:
                raise ValueError("Wrong weight range. First element must be smaller or equal to second element.")
            weights = np.array(weights)
            weights = weights/np.sum(weights)

        out_range = None
        duration = min(self.size/self.samplerate, other.size/self.samplerate)
        with self.cut(start_time=self._get_start() , end_time=self._get_start()+duration) as blend0:
            with other.cut(start_time=other._get_start(), end_time=other._get_start()+duration) as blend1:
                if normalize_first:
                    self_max = np.amax(self.array)
                    self_min = np.amin(self.array)
                    other_max = np.amax(other.array)
                    other_min = np.amin(other.array)
                    blend0 = (blend0-self_min)/(self_max-self_min)
                    blend1 = (blend1-other_min)/(other_max-other_min)

                    if new_range is None:
                        new_range = "left"

                right = min(blend0.size, blend1.size)
                sig = weights[0]*blend0[:right] + weights[1]*blend1[:right]
                metadata = {"blended_from": [self.to_dict(), other.to_dict()],
                            "normalize_first": normalize_first,
                            "array_cut": [0, right],
                            "time_cut": [0, duration]
                           }

                if not normalize_first:

                    return Audio(timeexp=1.0, array=sig, samplerate=self.samplerate, duration=duration, metadata=metadata)

                if isinstance(new_range, str):
                    out_range = OUT_RANGE[new_range](self_min, self_max, other_min, other_max)
                else:
                    out_range = new_range

                minsig = np.amin(sig)
                maxsig = np.amax(sig)
                sig = ((sig - minsig)/(maxsig-minsig))
                sig = (1-sig)*out_range[0] + sig*out_range[1]

                metadata["new_range"] = new_range
                window = TimeWindow(start=0, end=duration)

                return type(self)(timeexp=1.0,
                                  array=sig,
                                  window=window,
                                  samplerate=self.samplerate,
                                  duration=duration,
                                  metadata=metadata)

    def plot(self, ax=None, **kwargs):
        """Plot soundwave in the given axis."""
        ax = super().plot(ax=ax, **kwargs)

        array = self.array.copy()
        if 'vmax' in kwargs:
            maximum = np.abs(array).max()
            vmax = kwargs['vmax']
            array *= vmax / maximum

        if 'offset' in kwargs:
            array += kwargs['offset']

        ax.plot(
            self.times,
            array,
            c=kwargs.get('color', None),
            linewidth=kwargs.get('linewidth', 1),
            linestyle=kwargs.get('linestyle', None),
            alpha=kwargs.get('alpha', 1))

        return ax

    def to_dict(self):
        """Return a dictionary holding all audio metadata."""
        if self.media_info is None:
            media_info = None
        else:
            media_info = dict(self.media_info._asdict())

        return {
            'timeexp': self.timeexp,
            'media_info': media_info,
            'metadata': self.metadata.copy(),
            'id': self.id,
            **super().to_dict()
        }

    def __repr__(self):
        """Return a representation of the audio object."""
        data = OrderedDict()
        if self.path is not None:
            data['path'] = repr(self.path)
        else:
            data['array'] = repr(self.array)

        data['duration'] = self.duration
        data['samplerate'] = self.samplerate

        if self.timeexp != 1:
            data['timeexp'] = self.timeexp

        if not self._has_trivial_window():
            data['window'] = repr(self.window)

        args = [f'{key}={value}' for key, value in data.items()]
        args_string = ', '.join(args)
        return f'Audio({args_string})'
