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

    Parameters
    ----------
    audio : Audio
        Audio object associated to feature.
    """

    def __init__(
            self,
            audio=None,
            **kwargs):
        """Construct a feature.

        Parameters
        ----------
        audio : Audio
            Audio object associated to feature.

        """
        from yuntu.core.audio.audio import Audio
        if isinstance(audio, Audio):
            self._audio = audio
            self._audio_data = audio.to_dict()
        else:
            self._audio_data = audio

        super().__init__(**kwargs)

    @property
    def audio(self):
        """Audio object associated to feature."""
        if not hasattr(self, '_audio'):
            from yuntu.core.audio.audio import Audio
            data = self._audio_data.copy()
            data.pop('type')
            self._audio = Audio.from_dict(
                data,
                lazy=True)
        return self._audio

    def clean_audio(self):
        """Delete audio features's audio object."""
        del self._audio

    def _copy_dict(self, **kwargs):
        """Make a copy of a dictionary specification for the feature.
        Returns
        -------
        audio_meta : dict
            Dictionary holding feature specification.
        """
        return {
            'audio': self._audio_data,
            **super()._copy_dict(**kwargs),
        }

    def to_dict(self):
        """Generate feature specification as dictionary.

        Returns
        -------
        audio_meta : dict
            Dictionary holding feature specification.
        """
        data = super().to_dict()

        if self.has_audio():
            data['audio'] = self._audio_data

        return data

    def has_audio(self):
        """Return if this feature is linked to an Audio instance.

        Returns
        -------
        bool
            Wether the feature has an audio file link or not.
        """
        if not hasattr(self, '_audio_data'):
            return False

        return self._audio_data is not None

    def plot(self, ax=None, **kwargs):
        """Plot feature.

        Parameters
        ----------
        ax : matplotlib.axis
            An axis to plot graphics.

        Returns
        -------
        ax : matplotlib.axis
            Axis holding plot.

        """
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('audio', False):
            audio_kwargs = kwargs.get('audio_kwargs', {})
            self.audio.plot(ax=ax, **audio_kwargs)

        return ax

    def load(self, path=None):
        """Load feature.

        Parameters
        ----------
        path : str
            Any path holding media.

        Returns
        -------
        feature_array : numpy.array
            Array holding feature data.

        """
        result = super().load(path=path)
        self.clean_audio()
        return result

    def load_from_path(self, path=None):
        """Load feature from path.

        Parameters
        ----------
        path : str
            Any path holding feature as a '.npy' or '.npz' file.

        Returns
        -------
        feature_array : numpy.array
            Array holding feature data.

        """
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

    def write(self, path):  # pylint: disable=arguments-differ
        """Write the spectrogram matrix into the filesystem."""
        raise NotImplementedError("This feature has no write method.")


class TimeFeature(TimeMediaMixin, Feature):
    pass


class FrequencyFeature(FrequencyMediaMixin, Feature):
    pass


class TimeFrequencyFeature(TimeFrequencyMediaMixin, Feature):
    pass
