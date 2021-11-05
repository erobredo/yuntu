"""Spectrum class module."""
import numpy as np

from yuntu.core.audio.features.base import FrequencyFeature


class Spectrum(FrequencyFeature):
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
        return np.abs(np.fft.rfft(self.audio.array))

    def write(self, path=None):
        # TODO
        pass

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        ax.plot(
            self.array,
            self.frequencies,
            label=kwargs.get('label', None),
            color=kwargs.get('color', 'black'),
            linestyle=kwargs.get('linestyle', None),
            linewidth=kwargs.get('linewidth', 1))

        return ax
