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
    """Convert mel scale to Hz.

    Parameters
    ----------
    x : float
        Mel value.
    pos : float
        Unused position value.

    Returns
    -------
    str
        Formated label in Hz.
    """
    mel = mel_to_hz(x)
    return '%1.0f' % (mel)

class Spectrogram(TimeFrequencyFeature):
    """Class for spectrogram feature.

    Parameters
    ----------
    audio : Audio
        An audio object.
    n_fft : int
        Length of the windowed signal after padding with zeros.
    hop_length : int
        Number of audio samples between adjacent STFT columns.
    window_function : str, numpy.array, callable
        A window specification by name, array (of size n_fft) or function.
        Defaults to a raised cosine window (‘hann’), which is adequate for most
        applications in audio signal processing.
    max_freq : float
        Maximum frequency of the representation.
    freq_resolution : float
        Resolution of the frequency axis.
    frequency_axis : FrequencyAxis
        A specific frequency axis to take.
    duration : float
        Duration of the feature. This parameter is only considered when no
        audio is provided.
    time_resolution : float
        Resolution for time axis.
    time_axis : TimeAxis
        A specific time axis to take.
    array : numpy.array
        An array to use as feature's data.
    """
    plot_title = 'Amplitude Spectrogram'
    _default_cmap = 'gray'

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
        """Construct Spectrogram object.

        Parameters
        ----------
        audio : Audio
            An audio object.
        n_fft : int
            Length of the windowed signal after padding with zeros.
        hop_length : int
            Number of audio samples between adjacent STFT columns.
        window_function : str, numpy.array, callable
            A window specification by name, array (of size n_fft) or function.
            Defaults to a raised cosine window (‘hann’), which is adequate for most
            applications in audio signal processing.
        max_freq : float
            Maximum frequency of the representation.
        freq_resolution : float
            Resolution of the frequency axis.
        frequency_axis : FrequencyAxis
            A specific frequency axis to take.
        duration : float
            Duration of the feature. This parameter is only considered when no
            audio is provided.
        time_resolution : float
            Resolution for time axis.
        time_axis : TimeAxis
            A specific time axis to take.
        array : numpy.array
            An array to use as feature's data.
        """
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
        """Compute default resolution.

        Parameters
        ----------
        n_fft : int
            Lenght of signal to process by frame.
        max_freq : float
            Maximum frequency.

        Returns
        -------
        resolution : float
            Default resolution according to parameters.

        """
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
        """Apply transformation to array.

        Parameters
        ----------
        array : numpy.array
            Signal to transform.

        Returns
        -------
        np.array
            Computed transformation.

        """
        return np.abs(stft(array, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window_function))

    def compute(self):
        """Compute representation from audio data.

        Uses the spectrogram instance configurations for stft
        calculation.

        Returns
        -------
        np.array
            Computed representation of audio data.
        """
        if not self._has_trivial_window():
            start = self._get_start()
            end = self._get_end()
            array = self.audio.cut(
                start_time=start, end_time=end).array
        else:
            array = self.audio.array

        result = self.transform(array)

        max_index = self.get_index_from_frequency(self._get_max())
        min_index = self.get_index_from_frequency(self._get_min())

        start_index = self.get_index_from_time(self._get_start())
        end_index = self.get_index_from_time(self._get_end())

        cut_x = False
        cut_y = False

        if start_index != 0 or end_index != result.shape[1]:
            cut_x = True

        if min_index != 0 or max_index != result.shape[0]:
            cut_y = True

        if self._has_trivial_window() and not (cut_y or cut_x):
            return result

        slices = (
            slice(min_index, max_index),
            slice(start_index, end_index))

        return result[slices]

    def write(self, path):  # pylint: disable=arguments-differ
        """Write the spectrogram matrix into the filesystem."""
        # TODO

    @property
    def shape(self) -> Shape:
        """Get spectrogram shape.

        Returns
        -------
        Shape
            Spectrogram shape as object.

        """
        return Shape(rows=self.frequency_size, columns=self.time_size)

    @property
    def spectrum_values(self):
        """Get vertical axis values.

        Returns
        -------
        numpy.array
            Array of vertical values.

        """
        return self.frequencies

    def plot(self, ax=None, **kwargs):
        """Plot the spectrogram.

        Notes
        -----
        Will create a new figure if no axis (ax) was provided.

        Parameters
        ----------
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
            cmap=kwargs.get('cmap', self._default_cmap),
            alpha=kwargs.get('alpha', 1.0),
            shading=kwargs.get('shading', 'auto'))

        if kwargs.get('colorbar', False):
            plt.colorbar(mesh, ax=ax)

        return ax

    def iter_rows(self):
        """Iterate over spectrogram rows.

        Returns
        -------
        iterator
            Row iterator for feature values.

        """
        for row in self.array:
            yield row

    def iter_cols(self):
        """Iterate over spectrogram columns.

        Returns
        -------
        iterator
            Column iterator for feature values.

        """
        for col in self.array.T:
            yield col

    def power(self, lazy=False):
        """Get power spectrogram from spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.

        Returns
        -------
        PowerSpectrogram
            Power spectrogram computed from spectrogram values.

        """
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            kwargs['array'] = self.array ** 2

        return PowerSpectrogram(**kwargs)

    # pylint: disable=invalid-name
    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = 1.0,
            amin: Optional[float] = 1e-05,
            top_db: Optional[float] = 80.0):
        """Get decibel spectrogram from spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.

        Returns
        -------
        DecibelSpectrogram
            Decibel spectrogram computed from spectrogram values.

        """
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
        """Get mel spectrogram from spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.

        Returns
        -------
        MelSpectrogram
            Mel spectrogram computed from spectrogram values.

        """
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            max_freq = self.window.max
            sr = 2*(self.time_axis.resolution * self.hop_length // 2)
            n_mels = (1 + self.n_fft // 2)//4
            kwargs['array'] = melspectrogram(S=self.array**2,
                                             sr=sr,
                                             n_mels=n_mels)

        return MelSpectrogram(**kwargs)

    def to_dict(self):
        """Return feature's specification as dictionary.

        Returns
        -------
        dict
            A dictionary abstraction of the feature.

        """
        return {
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'window_function': self.window_function,
            **super().to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        """Produce instance from dictionary abstraction.

        Parameters
        ----------
        data : dict
            A dictionary holding configurations to generate this type of
            feature.

        Returns
        -------
        TimeFrequencyFeature
            One of the spectrogram features depending on input configs.

        """
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
    """Power spectrogram class.

    Accepts the same parameters as Spectrogram class.

    """
    plot_title = 'Power Spectrogram'

    def transform(self, array):
        """Apply transformation to array.

        Parameters
        ----------
        array : numpy.array
            Signal to transform.

        Returns
        -------
        np.array
            Computed transformation.

        """
        spectrogram = super().transform(array)
        return spectrogram**2

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.

        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = 1.0,
            amin: Optional[float] = 1e-05,
            top_db: Optional[float] = 80.0):
        """Get decibel spectrogram from power spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.
        ref : float
            Reference for amplitude scaling.
        amin : float
            Minimum treshold for amplitude.
        top_db : float
            Maximum decibels to represent.

        Returns
        -------
        DecibelSpectrogram
            Decibel spectrogram computed from power spectrogram values.

        """
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
            kwargs['array'] = power_to_db(self.array,
                                          ref=kwargs["ref"],
                                          amin=kwargs["amin"],
                                          top_db=kwargs['top_db'])

        return DecibelSpectrogram(**kwargs)

    def mel(self, lazy=False):
        """Get power spectrogram from spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.

        Returns
        -------
        MelSpectrogram
            Mel spectrogram computed from power spectrogram values.

        """
        kwargs = self._copy_dict()

        if not self.is_empty() or not lazy:
            max_freq = self.window.max
            sr = 2*(self.time_axis.resolution * self.hop_length // 2)
            n_mels = (1 + self.n_fft // 2)//4
            kwargs['array'] = melspectrogram(S=self.array, sr=sr,
                                             fmax=max_freq, n_mels=n_mels)

        return MelSpectrogram(**kwargs)

class DecibelSpectrogram(Spectrogram):
    """Decibel spectrogram class.

    Accepts all parameters for Spectrogram plus the following:

    Parameters
    ----------
    ref : float
        Reference for amplitude scaling.
    amin : float
        Minimum treshold for amplitude.
    top_db : float
        Maximum decibels to represent.
    """

    plot_title = 'Decibel Spectrogram'
    ref = 1.0
    amin = 1e-05
    top_db = 80.0

    def __init__(self, ref=None, amin=None, top_db=None, **kwargs):
        """Construct a decibel spectrogram.

        Accepts all parameters for Spectrogram __init__ plus the following:

        Parameters
        ----------
        ref : float
            Reference for amplitude scaling.
        amin : float
            Minimum treshold for amplitude.
        top_db : float
            Maximum decibels to represent.
        """
        if ref is not None:
            self.ref = ref

        if amin is not None:
            self.amin = amin

        if top_db is not None:
            self.top_db = top_db

        super().__init__(**kwargs)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.

        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.

        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(self, *args, **kwargs):
        """Get decibel spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.

        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def transform(self, array):
        """Apply transformation to array.

        Parameters
        ----------
        array : numpy.array
            Signal to transform.

        Returns
        -------
        np.array
            Computed transformation.
        """
        spectrogram = super().transform(array)
        return amplitude_to_db(
                spectrogram,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db)

class MelScaleAxis(FrequencyAxis):
    "An axis that has mel-scaled frequency."

    def get_bin(self, value):
        """Get bin associated to value.

        Parameters
        ----------
        value : float
            A value in Hz to convert.

        Returns
        -------
        bin : int
            Computed bin associated to value.

        """
        mel_value = hz_to_mel(value)
        return int(np.floor(mel_value * self.resolution))

    def get_bins(self, start=None, end=None, window=None, size=None):
        """Get values from bins."""
        start = self.get_start(start=start, window=window)
        end = self.get_end(end=end, window=window)

        mel_start = hz_to_mel(start)
        mel_end = hz_to_mel(end)

        if size is None:
            size = self.get_bin_nums(start, end)

        mel_freqs = np.linspace(mel_start, mel_end, size)

        return mel_to_hz(mel_freqs)

class MelSpectrogram(Spectrogram):
    """Mel spectrogram class.

    Accepts all parameters for Spectrogram class plus the following:

    Parameters
    ----------
    sr : float
        Sampling rate for mel computation.
    n_mels : int
        Number of Mel bands to generate
    """
    frequency_axis_class = MelScaleAxis
    plot_title = 'Mel-scaled Spectrogram'

    def __init__(self, *args, sr=None, n_mels=None, **kwargs):
        """Construct mel spectrogram.
        Accepts all parameters for Spectrogram __init__ plus the following:

        Parameters
        ----------
        sr : float
            Sampling rate for mel computation.
        n_mels : float
            Number of Mel bands to generate
        """
        self._sr = sr
        self._n_mels = n_mels
        super().__init__(*args,**kwargs)

    @property
    def sr(self):
        """Sample rate property for mel computation."""
        if self._sr is None:
            return 2*(self.time_axis.resolution * self.hop_length // 2)
        return self._sr

    @property
    def n_mels(self):
        """Number of bins in mel scale."""
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
        """Apply transformation to array.

        Parameters
        ----------
        array : numpy.array
            Signal to transform.

        Returns
        -------
        np.array
            Computed transformation.

        """
        power_spectrogram = super().transform(array)**2
        max_freq = self.window.max
        return melspectrogram(S=power_spectrogram, sr=self.sr, n_mels=self.n_mels)

    @property
    def spectrum_values(self):
        """Vertical axis values."""
        return hz_to_mel(self.frequencies)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.

        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.
        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(
            self,
            lazy: Optional[bool] = False,
            ref: Optional[float] = 1.0,
            amin: Optional[float] = 1e-05,
            top_db: Optional[float] = 80.0):
        """Get decibel spectrogram from power spec.

        Parameters
        ----------
        lazy : bool
            Wether to compute array right away (False) or not.
        ref : float
            Reference for amplitude scaling.
        amin : float
            Minimum treshold for amplitude.
        top_db : float
            Maximum decibels to represent.

        Returns
        -------
        DecibelMelSpectrogram
            Decibel mel spectrogram computed from power spectrogram values.

        """
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
            kwargs['array'] = power_to_db(self.array,
                                          ref=kwargs["ref"],
                                          amin=kwargs["amin"],
                                          top_db=kwargs['top_db'])
        return DecibelMelSpectrogram(**kwargs)

    def plot(self, ax=None, **kwargs):
        """Plot feature.
        Parameters
        ----------
        ax : matplotlib.Axes
            An axes to plot

        Returns
        -------
        ax : matplotlib.Axes
            Axes with plot.

        """
        ax = super().plot(ax, **kwargs)
        formatter = FuncFormatter(label_to_hz)
        ax.yaxis.set_major_formatter(formatter)

        return ax

class DecibelMelSpectrogram(MelSpectrogram):
    """Decibel mel spectrogram class.

    Accepts all parameters for MelSpectrogram plus the following:

    Parameters
    ----------
    ref : float
        Reference for amplitude scaling.
    amin : float
        Minimum treshold for amplitude.
    top_db : float
        Maximum decibels to represent.
    """
    plot_title = 'Decibel Mel-scaled Spectrogram'
    ref = 1.0
    amin = 1e-05
    top_db = 80.0

    def __init__(self, ref=None, amin=None, top_db=None, **kwargs):
        """Construct a decibel mel-spectrogram.

        Accepts all parameters for MelSpectrogram __init__ plus the following:

        Parameters
        ----------
        ref : float
            Reference for amplitude scaling.
        amin : float
            Minimum treshold for amplitude.
        top_db : float
            Maximum decibels to represent.
        """
        if ref is not None:
            self.ref = ref

        if amin is not None:
            self.amin = amin

        if top_db is not None:
            self.top_db = top_db

        super().__init__(**kwargs)

    def mel(self, *args, **kwargs):
        """Get mel spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.
        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def power(self, *args, **kwargs):
        """Get power spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.
        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def db(self, *args, **kwargs):
        """Get decibel spectrogram from spec.

        Raises
        -------
        NotImplementedError
            Always, method not implemented.
        """
        raise NotImplementedError("The method is not implemented for this feature.")

    def transform(self, array):
        """Apply transformation to array.

        Parameters
        ----------
        array : numpy.array
            Signal to transform.

        Returns
        -------
        np.array
            Computed transformation.

        """
        mel_spectrogram = super().transform(array)
        return power_to_db(
                mel_spectrogram,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db)
