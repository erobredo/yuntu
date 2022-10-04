"""Spectrum class module."""
import numpy as np

from yuntu.core.audio.features.base import FrequencyFeature


class Spectrum(FrequencyFeature):
    """A class for spectral features.

    Parameters
    ----------
    audio : Audio
        An audio object.
    array : numpy.array
        An array to use as feature's data.
    min_freq : float
        Minimum frequency of the representation.
    max_freq : float
        Maximum frequency of the representation.
    resolution : float
        Frequency resolution.
    frequency_axis : FrequencyAxis
        A specific frequency axis to take.
    """
    plot_title = 'Spectrum'

    def __init__(
            self,
            audio=None,
            array=None,
            min_freq=0,
            max_freq=None,
            resolution=None,
            frequency_axis=None,
            **kwargs):
        """Construct spectral feature.

        Parameters
        ----------
        audio : Audio
            An audio object.
        array : numpy.array
            An array to use as feature's data.
        min_freq : float
            Minimum frequency of the representation.
        max_freq : float
            Maximum frequency of the representation.
        resolution : float
            Frequency resolution.
        frequency_axis : FrequencyAxis
            A specific frequency axis to take.
        """

        if frequency_axis is None:
            from yuntu.core.audio.audio import Audio
            if audio is not None and not isinstance(audio, Audio):
                audio = Audio.from_dict(audio)

            if max_freq is None:
                if array is not None and resolution is not None:
                    max_freq = len(array) / resolution
                elif audio is not None:
                    max_freq = audio.samplerate / 2
                else:
                    message = (
                        'If no audio is provided a max_freq must be set')
                    raise ValueError(message)

            if resolution is None:
                if array is not None:
                    length = len(array)
                elif audio is not None:
                    length = len(audio)
                    if length % 2:
                        length = (length + 1) / 2
                    else:
                        length = (length / 2) + 1
                else:
                    message = (
                        'If no audio or array is provided a '
                        'resolution must be set')
                    raise ValueError(message)

                resolution = length / max_freq

        super().__init__(
            audio=audio,
            min_freq=min_freq,
            max_freq=max_freq,
            resolution=resolution,
            frequency_axis=frequency_axis,
            array=array,
            **kwargs)

    def compute(self):
        """Compute representation from audio data.

        Uses the spectrogram instance configurations for stft
        calculation.

        Returns
        -------
        np.array
            Computed representation of audio data.
        """
        return np.abs(np.fft.rfft(self.audio.array))

    def write(self, path=None):
        """Write feature to path.

        Parameters
        ----------
        path : str
            Path to write feature.

        Raises
        ------
        NotImplementedError
            Always, not implemented
        """
        NotImplementedError("Write method not implemented for this feature.")

    def plot(self, ax=None, **kwargs):
        """Plot feature.

        Notes
        -----
        Will create a new figure if no axis (ax) was provided.

        Parameters
        ----------
        ax : matplotlib.Axes
            Where to plot. If not provided, a new axis will be created.

        Returns
        -------
        matplotlib.Axes
            The axis where the feature was plotted.

        """
        ax = super().plot(ax=ax, **kwargs)

        ax.plot(
            self.array,
            self.frequencies,
            label=kwargs.get('label', None),
            color=kwargs.get('color', 'black'),
            linestyle=kwargs.get('linestyle', None),
            linewidth=kwargs.get('linewidth', 1))

        return ax
