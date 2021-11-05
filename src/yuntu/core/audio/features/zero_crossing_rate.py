"""Zero crossing class module."""
import librosa
import numpy as np

from yuntu.core.audio.features.base import TimeFeature


THRESHOLD = 1e-10
FRAME_LENGTH = 2048
HOP_LENGTH = 512


class ZeroCrossingRate(TimeFeature):
    plot_title = 'Zero Crossing Rate'

    def __init__(
            self,
            audio=None,
            array=None,
            threshold=THRESHOLD,
            ref_magnitude=None,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            duration=None,
            resolution=None,
            time_axis=None,
            **kwargs):

        self.threshold = threshold
        self.ref_magnitude = ref_magnitude
        self.frame_length = frame_length
        self.hop_length = hop_length
        from yuntu.core.audio.audio import Audio
        if audio is not None and not isinstance(audio, Audio):
            audio = Audio.from_dict(audio)

        if duration is None:
            if audio is None:
                message = (
                    'If no audio is provided a duration must be set')
                raise ValueError(message)
            duration = audio.duration

        if resolution is None:
            if array is not None:
                length = len(array)
            elif audio is not None:
                length = 1 + (len(audio) - frame_length) // hop_length
            else:
                message = (
                    'If no audio or array is provided a samplerate must be '
                    'set')
                raise ValueError(message)

            resolution = length / duration

        super().__init__(
            audio=audio,
            duration=duration,
            resolution=resolution,
            array=array,
            time_axis=time_axis,
            **kwargs)

    def compute(self):
        zero_crossings = librosa.core.zero_crossings(
            self.audio,
            threshold=self.threshold,
            ref_magnitude=self.ref_magnitude)

        frames = librosa.util.frame(
            zero_crossings,
            frame_length=self.frame_length,
            hop_length=self.hop_length)

        frame_duration = self.frame_length / self.audio.samplerate
        crossings_per_frame = frames.sum(axis=-2)
        return crossings_per_frame / (2 * frame_duration)

    def to_dict(self):
        return {
            'threshold': self.threshold,
            'ref_magnitude': self.ref_magnitude,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            **super().to_dict(),
        }

    def write(self, path=None):
        # TODO
        pass

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        masked = np.ma.masked_less_equal(
            self.array,
            kwargs.get('min_freq', 0))

        if kwargs.get('max_freq', 0):
            masked = np.ma.masked_greater_equal(
                self.array,
                kwargs.get('max_freq'))

        ax.plot(
            self.times,
            masked,
            color=kwargs.get('color', 'black'),
            linestyle=kwargs.get('linestyle', 'dotted'),
            linewidth=kwargs.get('linewidth', 1))

        return ax
