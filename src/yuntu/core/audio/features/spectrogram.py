"""Spectrogram class module."""
from typing import Optional
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from librosa import mel_frequencies, hz_to_mel, mel_to_hz
from librosa.core import amplitude_to_db
from librosa.core import power_to_db
from librosa.feature import melspectrogram

import yuntu.core.windows as windows
from yuntu.core.axis import FrequencyAxis
from yuntu.core.audio.features.base import TimeFrequencyFeature
from yuntu.core.audio.features.utils import stft


BOXCAR = 'boxcar'
TRIANG = 'triang'
BLACKMAN = 'blackman'
HAMMING = 'hamming'
HANN = 'hann'
BARTLETT = 'bartlett'
FLATTOP = 'flattop'
PARZEN = 'parzen'
BOHMAN = 'bohman'
BLACKMANHARRIS = 'blackmanharris'
NUTTALL = 'nuttall'
BARTHANN = 'barthann'
WINDOW_FUNCTIONS = [
    BOXCAR,
    TRIANG,
    BLACKMAN,
    HAMMING,
    HANN,
    BARTLETT,
    FLATTOP,
    PARZEN,
    BOHMAN,
    BLACKMANHARRIS,
    NUTTALL,
    BARTHANN,
]

N_FFT = 1024
HOP_LENGTH = 512
WINDOW_FUNCTION = HANN

Shape = namedtuple('Shape', ['rows', 'columns'])

def label_to_hz(x, pos):
    'The two args are the value and tick position'
    mel = mel_to_hz(x)
    return '%1.0f' % (mel)

class Spectrogram(TimeFrequencyFeature):
    """Spectrogram class."""
    plot_title = 'Amplitude Spectrogram'

    def __init__(
            self,
            audio=None,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window_function=WINDOW_FUNCTION,
            max_freq=None,
            freq_resolution=None,
            frequency_axis=None,
            duration=None,
            time_resolution=None,
            time_axis=None,
            array=None,
            **kwargs):
        """Construct Spectrogram object."""
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_function = window_function

        if time_axis is None:
            from yuntu.core.audio.audio import Audio
            if audio is not None and not isinstance(audio, Audio):
                audio = Audio.from_dict(audio)

            if duration is None:
                if audio is None:
                    message = (
                        'If no audio is provided a duration must be set')
                    raise ValueError(message)
                duration = audio.duration

            if time_resolution is None:
                if array is not None:
                    columns = array.shape[self.frequency_axis_index]
                elif audio is not None:
                    columns = 1 + len(audio) // hop_length
                else:
                    message = (
                        'If no audio or array is provided a samplerate must be '
                        'set')
                    raise ValueError(message)

                time_resolution = columns / duration

        if frequency_axis is None:
            if max_freq is None:
                if audio is not None:
                    if isinstance(audio, dict):
                        samplerate = audio["media_info"]["samplerate"]
                    else:
                        samplerate = audio.samplerate
                    max_freq = samplerate / 2
                else:
                    max_freq = time_resolution * hop_length // 2
            if freq_resolution is None:
                freq_resolution = self._default_resolution(n_fft, max_freq)

        super().__init__(
            audio=audio,
            duration=duration,
            time_resolution=time_resolution,
            max_freq=max_freq,
            freq_resolution=freq_resolution,
            frequency_axis=frequency_axis,
            array=array,
            time_axis=time_axis,
            **kwargs)


    @staticmethod
    def _default_resolution(n_fft, max_freq):
        rows = 1 + n_fft // 2
        return rows / max_freq

    def __repr__(self):
        data = OrderedDict()

        if self.n_fft != N_FFT:
            data['n_fft'] = self.n_fft

        if self.hop_length != HOP_LENGTH:
            data['hop_length'] = self.hop_length

        if self.window_function != WINDOW_FUNCTION:
            data['window_function'] = self.window_function

        has_path = self.path_exists()
        if has_path:
            data['path'] = repr(self.path)

        has_audio = self.has_audio()
        if not has_path and has_audio:
            data['audio'] = f'Audio(path={self._audio_data["path"]})'

        if not has_audio and not has_path:
            data['array'] = repr(self.array)

        if not self._has_trivial_window():
            data['window'] = repr(self.window)

        class_name = type(self).__name__
        args = [f'{key}={value}' for key, value in data.items()]
        args_string = ', '.join(args)

        return f'{class_name}({args_string})'

    def transform(self, array):
        """Apply transformation to array"""
        return np.abs(stft(array, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window_function))

    def compute(self):
        """Compute spectrogram from audio data.

        Uses the spectrogram instance configurations for stft
        calculation.

        Returns
        -------
        np.array
            Calculated spectrogram.
        """
        if not self._has_trivial_window():
            start = self._get_start()
            end = self._get_end()
            array = self.audio.cut(
                start_time=start, end_time=end).array
        else:
            array = self.audio.array

        result = self.transform(array)

        if self._has_trivial_window():
            return result

        max_index = self.get_index_from_frequency(self._get_max())
        min_index = self.get_index_from_frequency(self._get_min())

        start_index = self.get_index_from_time(self._get_start())
        end_index = self.get_index_from_time(self._get_end())

        slices = (
            slice(min_index, max_index),
            slice(start_index, end_index))

        return result[slices]

    def write(self, path):  # pylint: disable=arguments-differ
        """Write the spectrogram matrix into the filesystem."""
        # TODO

    @property
    def shape(self) -> Shape:
        """Get spectrogram shape."""
        return Shape(rows=self.frequency_size, columns=self.time_size)

    @property
    def spectrum_values(self):
        return self.frequencies

    def plot(self, ax=None, **kwargs):
        """Plot the spectrogram.

        Notes
        -----
        Will create a new figure if no axis (ax) was provided.

        Arguments
        ---------
        figsize: tuple, optional
            Figure size in inches
        cmap: str, optional
            Colormap to use for spectrogram plotting
        colorbar: bool, optional
            Flag indicating whether to draw a colorbar.
        w_xlabel: bool, optional
            Flag indicating wheter to set the x-axis label of
            the provided axis.
        xlabel: str, optional
            The label to use for the x-axis. Defaults to "Time(s)".
        w_ylabel: bool, optional
            Flag indicating wheter to set the y-axis label of
            the provided axis.
        ylabel: str, optional
            The label to use for the y-axis. Defaults to "Frequency(Hz)".
        w_title: bool, optional
            Flag indicating wheter to set the title of
            the provided axis.
        title: str, optional
            The title to use. Defaults to "Spectrogram".

        Returns
        -------
        matplotlib.Axes
            The axis into which the spectrogram was plotted.
        """
        ax = super().plot(ax=ax, **kwargs)

        spectrogram = self.array
        minimum = spectrogram.min()
        maximum = spectrogram.max()

        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)

        if 'pvmin' in kwargs:
            pvmin = kwargs['pvmin']
            vmin = minimum + (maximum - minimum) * pvmin

        if 'pvmax' in kwargs:
            pvmax = kwargs['pvmax']
            vmax = minimum + (maximum - minimum) * pvmax

        mesh = ax.pcolormesh(
            self.times,
            self.spectrum_values,
            spectrogram,
            vmin=vmin,
            vmax=vmax,
            cmap=kwargs.get('cmap', 'gray'),
            alpha=kwargs.get('alpha', 1.0),
            shading=kwargs.get('shading', 'auto'))

        if kwargs.get('colorbar', False):
            plt.colorbar(mesh, ax=ax)

        return ax

    def iter_rows(self):
        """Iterate over spectrogram rows."""
        for row in self.array:
            yield row

    def iter_cols(self):
        """Iterate over spectrogram columns."""
        for col in self.array.T:
            yield col

    def power(self, lazy=False):
        """Get power spectrogram from spec."""
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            kwargs['array'] = self.array ** 2

        return PowerSpectrogram(**kwargs)

    # pylint: disable=invalid-name
    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = None,
            amin: Optional[float] = None,
            top_db: Optional[float] = None):
        """Get decibel spectrogram from spec."""
        kwargs = {
            'ref': ref,
            'amin': amin,
            'top_db': top_db,
            **self._copy_dict()
        }

        if not self.is_empty() or not lazy:
            kwargs['array'] = amplitude_to_db(self.array)

        return DecibelSpectrogram(**kwargs)

    def mel(self, lazy=False):
        """Get power spectrogram from spec."""
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            max_freq = self.window.max
            sr = 2*(self.time_axis.resolution * self.hop_length // 2)
            n_mels = (1 + self.n_fft // 2)//4
            kwargs['array'] = melspectrogram(S=self.array**2, sr=sr,
                                             fmax=max_freq, n_mels=n_mels)

        return MelSpectrogram(**kwargs)

    def to_dict(self):
        """Return spectrogram metadata."""
        return {
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'window_function': self.window_function,
            **super().to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        spec_dict = dict(data)
        spec_type = spec_dict.pop('type', None)

        if spec_type == 'Spectrogram':
            return Spectrogram(**spec_dict)

        if spec_type == 'PowerSpectrogram':
            return PowerSpectrogram(**spec_dict)

        if spec_type == 'DecibelSpectrogram':
            return DecibelSpectrogram(**spec_dict)

        if spec_type == 'MelSpectrogram':
            return MelSpectrogram(**spec_dict)

        if spec_type == 'DecibelMelSpectrogram':
            return DecibelMelSpectrogram(**spec_dict)

        raise ValueError('Unknown or missing units')


class PowerSpectrogram(Spectrogram):
    """Power spectrogram class."""
    plot_title = 'Power Spectrogram'

    def transform(self, array):
        """Apply transformation to array"""
        spectrogram = super().transform(array)
        return spectrogram**2

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = None,
            amin: Optional[float] = None,
            top_db: Optional[float] = None):
        """Get decibel spectrogram from power spec."""
        kwargs = self.to_dict()
        kwargs['annotations'] = self.annotations
        kwargs['window'] = self.window

        if ref is not None:
            kwargs['ref'] = ref

        if amin is not None:
            kwargs['amin'] = amin

        if top_db is not None:
            kwargs['top_db'] = top_db

        if self.has_audio():
            kwargs['audio'] = self.audio

        if not self.is_empty() or not lazy:
            kwargs['array'] = power_to_db(self.array)

        return DecibelSpectrogram(**kwargs)

    def mel(self, lazy=False):
        """Get power spectrogram from spec."""
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            max_freq = self.window.max
            sr = 2*(self.time_axis.resolution * self.hop_length // 2)
            n_mels = (1 + self.n_fft // 2)//4
            kwargs['array'] = melspectrogram(S=self.array, sr=sr,
                                             fmax=max_freq, n_mels=n_mels)

        return MelSpectrogram(**kwargs)

class DecibelSpectrogram(Spectrogram):
    """Decibel spectrogram class."""
    plot_title = 'Decibel Spectrogram'
    ref = 1.0
    amin = 1e-05
    top_db = 80.0

    def __init__(self, ref=None, amin=None, top_db=None, **kwargs):
        """Construct a decibel spectrogram."""
        if ref is not None:
            self.ref = ref

        if amin is not None:
            self.amin = amin

        if top_db is not None:
            self.top_db = top_db

        super().__init__(**kwargs)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(self, *args, **kwargs):
        """Get decibel spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def transform(self, array):
        """Apply transformation to array"""
        spectrogram = super().transform(array)
        return amplitude_to_db(
                spectrogram,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db)

class MelScaleAxis(FrequencyAxis):
    "An axis that has mel-scaled frequency."

    def get_bin(self, value):
        mel_value = hz_to_mel(value)
        return int(np.floor(mel_value * self.resolution))

    def get_bins(self, start=None, end=None, window=None, size=None):
        start = self.get_start(start=start, window=window)
        end = self.get_end(end=end, window=window)

        mel_start = hz_to_mel(start)
        mel_end = hz_to_mel(end)

        if size is None:
            size = self.get_bin_nums(start, end)

        mel_freqs = np.linspace(mel_start, mel_end, size)

        return mel_to_hz(mel_freqs)

class MelSpectrogram(Spectrogram):
    """Decibel spectrogram class."""
    frequency_axis_class = MelScaleAxis
    plot_title = 'Mel-scaled Spectrogram'

    def __init__(self, *args, sr=None, n_mels=None, **kwargs):
        self._sr = sr
        self._n_mels = n_mels
        super().__init__(*args,**kwargs)

    @property
    def sr(self):
        if self._sr is None:
            return 2*(self.time_axis.resolution * self.hop_length // 2)
        return self._sr

    @property
    def n_mels(self):
        if self._n_mels is None:
            self._n_mels = (1 + self.n_fft // 2)//4
        return self._n_mels

    @staticmethod
    def _default_resolution(n_fft, max_freq):
        """Default resolution for feature according to parameters"""
        rows = (1 + n_fft // 2)//4
        max_mel = hz_to_mel(max_freq)
        return rows / max_mel

    def transform(self, array):
        """Apply transformation to array"""
        power_spectrogram = super().transform(array)**2
        max_freq = self.window.max
        return melspectrogram(S=power_spectrogram, sr=self.sr, fmax=max_freq, n_mels=self.n_mels)

    @property
    def spectrum_values(self):
        return hz_to_mel(self.frequencies)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = None,
            amin: Optional[float] = None,
            top_db: Optional[float] = None):
        """Get decibel spectrogram from power spec."""
        kwargs = self.to_dict()
        kwargs['annotations'] = self.annotations
        kwargs['window'] = self.window

        if ref is not None:
            kwargs['ref'] = ref

        if amin is not None:
            kwargs['amin'] = amin

        if top_db is not None:
            kwargs['top_db'] = top_db

        if self.has_audio():
            kwargs['audio'] = self.audio

        if not self.is_empty() or not lazy:
            kwargs['array'] = power_to_db(self.array)

        return DecibelMelSpectrogram(**kwargs)

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax, **kwargs)
        formatter = FuncFormatter(label_to_hz)
        ax.yaxis.set_major_formatter(formatter)

        return ax

class DecibelMelSpectrogram(MelSpectrogram):
    """Decibel spectrogram class."""
    plot_title = 'Decibel Mel-scaled Spectrogram'
    ref = 1.0
    amin = 1e-05
    top_db = 80.0

    def __init__(self, ref=None, amin=None, top_db=None, **kwargs):
        """Construct a decibel mel-spectrogram."""
        if ref is not None:
            self.ref = ref

        if amin is not None:
            self.amin = amin

        if top_db is not None:
            self.top_db = top_db

        super().__init__(**kwargs)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(self, *args, **kwargs):
        """Get decibel spectrogram from spec."""
        raise NotImplementedError("The method is not implemented for this feature.")

    def transform(self, array):
        """Apply transformation to array"""
        mel_spectrogram = super().transform(array)
        return power_to_db(
                mel_spectrogram,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db)
