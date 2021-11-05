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

        if samplerate is None and duration is None and array is not None:
            raise ValueError("When creating audio from numpy arrays," +
                             "either samplerate or duration must be specified.")

        if samplerate is not None and duration is None and array is not None:
            duration = len(array)/samplerate
        if samplerate is None and duration is not None and array is not None:
            samplerate = np.round(float(len(array))/duration).astype(int)

        if ((samplerate is None) or (duration is None)) and media_info is None:
            media_info = self.read_info()

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
        return self.window.end

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
