"""Feature class module."""
import os

import numpy as np
from yuntu.core.media.base import Media
from yuntu.core.media.time import TimeMediaMixin
from yuntu.core.media.frequency import FrequencyMediaMixin
from yuntu.core.media.time_frequency import TimeFrequencyMediaMixin


# TODO: Fix load after cut!!!


# pylint: disable=abstract-method
class Feature(Media):
    """Feature base class.

    This is the base class for all audio features. A feature contains
    information extracted from the audio data.
    """

    def __init__(
            self,
            audio=None,
            **kwargs):
        """Construct a feature."""
        from yuntu.core.audio.audio import Audio
        if isinstance(audio, Audio):
            self._audio = audio
            self._audio_data = audio.to_dict()
        else:
            self._audio_data = audio

        super().__init__(**kwargs)

    @property
    def audio(self):
        if not hasattr(self, '_audio'):
            from yuntu.core.audio.audio import Audio
            data = self._audio_data.copy()
            data.pop('type')
            self._audio = Audio.from_dict(
                data,
                lazy=True)
        return self._audio

    def clean_audio(self):
        del self._audio

    def _copy_dict(self, **kwargs):
        return {
            'audio': self._audio_data,
            **super()._copy_dict(**kwargs),
        }

    def to_dict(self):
        data = super().to_dict()

        if self.has_audio():
            data['audio'] = self._audio_data

        return data

    def has_audio(self):
        """Return if this feature is linked to an Audio instance."""
        if not hasattr(self, '_audio_data'):
            return False

        return self._audio_data is not None

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('audio', False):
            audio_kwargs = kwargs.get('audio_kwargs', {})
            self.audio.plot(ax=ax, **audio_kwargs)

        return ax

    def load(self, path=None):
        result = super().load(path=path)
        self.clean_audio()
        return result

    def load_from_path(self, path=None):
        if path is None:
            path = self.path

        extension = os.path.splitext(path)[1]
        if extension == 'npy':
            try:
                return np.load(self.path)
            except IOError:
                message = (
                    'The provided path for this feature object could '
                    f'not be read. (path={self.path})')
                raise ValueError(message)

        if extension == 'npz':
            try:
                with np.load(self.path) as data:
                    return data[type(self).__name__]
            except IOError:
                message = (
                    'The provided path for this feature object could '
                    f'not be read. (path={self.path})')
                raise ValueError(message)

        message = (
            'The provided path does not have a numpy file extension. '
            f'(extension={extension})')
        raise ValueError(message)


class TimeFeature(TimeMediaMixin, Feature):
    pass


class FrequencyFeature(FrequencyMediaMixin, Feature):
    pass


class TimeFrequencyFeature(TimeFrequencyMediaMixin, Feature):
    pass
