"""Audio Feature module."""
from typing import Optional

import yuntu.core.audio.features.spectrogram as spectrogram
import yuntu.core.audio.features.zero_crossing_rate as zcr
import yuntu.core.audio.features.spectrum as spectrum


class AudioFeatures:
    """Audio Features class.

    This class is syntactic sugar to access all available features
    that can be derived from an Audio object.

    Construct the Audio Feature object.
    Parameters
    ----------
    audio : Audio
        Audio file associated to features.
    """

    spectrogram_class = spectrogram.Spectrogram
    power_spectrogram_class = spectrogram.PowerSpectrogram
    db_spectrogram_class = spectrogram.DecibelSpectrogram
    mel_spectrogram_class = spectrogram.MelSpectrogram
    db_mel_spectrogram_class = spectrogram.DecibelMelSpectrogram

    def __init__(self, audio):
        """Construct the Audio Feature object.
        Parameters
        ----------
        audio : Audio
            Audio file associated to features.
        """
        self.audio = audio

    @staticmethod
    def list():
        """Get available features.

        Returns
        -------
        feature_names : list
            A list of available feature names.
        """
        return [
            'spectrogram',
            'power_spectrogram',
            'db_spectrogram'
            'mel_spectrogram',
        ]

    def get_base_kwargs(self):
        """Get basic kwargs.

        Returns
        -------
        basic_kwargs : dict
            A dictionary containing basic named arguments for this feature.
        """
        return {
            'window': self.audio.window,
            'annotations': self.audio.annotations
        }

    def spectrum(self, lazy=False):
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy
        return spectrum.Spectrum(audio=self.audio, **kwargs)

    def spectrogram(
            self,
            n_fft: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_function: Optional[str] = None,
            lazy: Optional[bool] = False,
            max_freq: Optional[float] = None):
        """Get amplitude spectrogram.

        Parameters
        ----------
        n_fft : int
            Size of input signal for STFT.
        hop_length : int
            Frame overlap for STFT.
        window_function : str
            Name of window function to apply in STFT.
        lazy : bool
            Wether to compute feature right away or wait until needed (True).
        max_freq : float
            Maximum frequency to register in feature.

        Returns
        -------
        spectrogram : Spectrogram
            Amplitude spectrogram.

        """
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy
        if n_fft is not None:
            kwargs['n_fft'] = n_fft

        if hop_length is not None:
            kwargs['hop_length'] = hop_length

        if window_function is not None:
            kwargs['window_function'] = window_function

        if max_freq is not None:
            kwargs['max_freq'] = max_freq

        return self.spectrogram_class(audio=self.audio, **kwargs)

    def power_spectrogram(
            self,
            n_fft: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_function: Optional[str] = None,
            lazy: Optional[bool] = False,
            max_freq: Optional[float] = None):
        """Get power spectrogram.

        Parameters
        ----------
        n_fft : int
            Size of input signal for STFT.
        hop_length : int
            Frame overlap for STFT.
        window_function : str
            Name of window function to apply in STFT.
        lazy : bool
            Wether to compute feature right away or wait until needed (True).
        max_freq : float
            Maximum frequency to register in feature.

        Returns
        -------
        power_spectrogram : PowerSpectrogram
            Power spectrogram.

        """
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy
        if n_fft is not None:
            kwargs['n_fft'] = n_fft

        if hop_length is not None:
            kwargs['hop_length'] = hop_length

        if window_function is not None:
            kwargs['window_function'] = window_function

        if max_freq is not None:
            kwargs['max_freq'] = max_freq

        return self.power_spectrogram_class(audio=self.audio, **kwargs)

    def db_spectrogram(
            self,
            n_fft: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_function: Optional[str] = None,
            lazy: Optional[bool] = False,
            ref: Optional[float] = None,
            amin: Optional[float] = None,
            top_db: Optional[float] = None,
            max_freq: Optional[float] = None):
        """Get decibel spectrogram.

        Parameters
        ----------
        n_fft : int
            Size of input signal for STFT.
        hop_length : int
            Frame overlap for STFT.
        window_function : str
            Name of window function to apply in STFT.
        lazy : bool
            Wether to compute feature right away or wait until needed (True).
        ref : float
            Reference for computations.
        amin : float
            Min amplitude for computations.
        top_db : float
            Maximum decibels for computation.
        max_freq : float
            Maximum frequency to register in feature.

        Returns
        -------
        db_spectrogram : DecibelSpectrogram
            Decibel spectrogram.

        """
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy

        if n_fft is not None:
            kwargs['n_fft'] = n_fft

        if hop_length is not None:
            kwargs['hop_length'] = hop_length

        if window_function is not None:
            kwargs['window_function'] = window_function

        if ref is not None:
            kwargs['ref'] = ref

        if amin is not None:
            kwargs['amin'] = amin

        if top_db is not None:
            kwargs['top_db'] = top_db

        if max_freq is not None:
            kwargs['max_freq'] = max_freq

        return self.db_spectrogram_class(audio=self.audio, **kwargs)

    def mel_spectrogram(
            self,
            n_fft: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_function: Optional[str] = None,
            lazy: Optional[bool] = False,
            max_freq: Optional[float] = None,
            n_mels: Optional[int] = None,
            sr: Optional[int] = None):
        """Get mel spectrogram.

        Parameters
        ----------
        n_fft : int
            Size of input signal for STFT.
        hop_length : int
            Frame overlap for STFT.
        window_function : str
            Name of window function to apply in STFT.
        lazy : bool
            Wether to compute feature right away or wait until needed (True).
        max_freq : float
            Maximum frequency to register in feature.
        n_mels : int
            Number of mel bands to generate.
        sr : int
            Samplerate of signal.

        Returns
        -------
        mel_spectrogram : MelSpectrogram
            Mel spectrogram.

        """
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy
        if n_fft is not None:
            kwargs['n_fft'] = n_fft

        if hop_length is not None:
            kwargs['hop_length'] = hop_length

        if window_function is not None:
            kwargs['window_function'] = window_function

        if n_mels is not None:
            kwargs['n_mels'] = n_mels

        if sr is not None:
            kwargs['sr'] = sr

        if max_freq is not None:
            kwargs['max_freq'] = max_freq

        return self.mel_spectrogram_class(audio=self.audio, **kwargs)

    def db_mel_spectrogram(
            self,
            n_fft: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_function: Optional[str] = None,
            lazy: Optional[bool] = False,
            ref: Optional[float] = None,
            amin: Optional[float] = None,
            top_db: Optional[float] = None,
            max_freq: Optional[float] = None,
            n_mels: Optional[int] = None,
            sr: Optional[int] = None):
        """Get decibel mel spectrogram.

        Parameters
        ----------
        n_fft : int
            Size of input signal for STFT.
        hop_length : int
            Frame overlap for STFT.
        window_function : str
            Name of window function to apply in STFT.
        lazy : bool
            Wether to compute feature right away or wait until needed (True).
        ref : float
            Reference for computations.
        amin : float
            Min amplitude for computations.
        top_db : float
            Maximum decibels for computation.
        max_freq : float
            Maximum frequency to register in feature.
        n_mels : int
            Number of mel bands to generate.
        sr : int
            Samplerate of signal.

        Returns
        -------
        db_mel_spectrogram : DecibelMelSpectrogram
            Mel spectrogram in decibels.

        """
        kwargs = self.get_base_kwargs()
        kwargs['lazy'] = lazy
        if n_fft is not None:
            kwargs['n_fft'] = n_fft

        if hop_length is not None:
            kwargs['hop_length'] = hop_length

        if window_function is not None:
            kwargs['window_function'] = window_function

        if n_mels is not None:
            kwargs['n_mels'] = n_mels

        if sr is not None:
            kwargs['sr'] = sr

        if ref is not None:
            kwargs['ref'] = ref

        if amin is not None:
            kwargs['amin'] = amin

        if top_db is not None:
            kwargs['top_db'] = top_db

        if max_freq is not None:
            kwargs['max_freq'] = max_freq

        return self.db_mel_spectrogram_class(audio=self.audio, **kwargs)

    def zcr(
            self,
            threshold=zcr.THRESHOLD,
            ref_magnitude=None,
            frame_length=zcr.FRAME_LENGTH,
            hop_length=zcr.HOP_LENGTH,
            **kwargs):
        """Get zero crossing rate.

        Parameters
        ----------
        threshold : float
            If specified, values where -threshold <= y <= threshold are clipped
            to 0 (see librosa.zero_crossings documentation.)
        ref_magnitude : float, callable
            Scale threshold to this magnitude or callable.
        frame_length : int
            Frame size for individual zcr computations.
        hop_length : int
            Overlap between frames for zcr.

        Returns
        -------
        zcr : ZeroCrossingRate
            Zero crossing rate by frames.

        """
        kwargs = {
            'threshold': threshold,
            'ref_magnitude': ref_magnitude,
            'frame_length': frame_length,
            'hop_length': hop_length,
            **kwargs,
            **self.get_base_kwargs()
        }

        return zcr.ZeroCrossingRate(audio=self.audio, **kwargs)
