"""Tools for audio transformation."""
import numpy as np
import librosa


def stft(signal,
         n_fft,
         hop_length,
         win_length=None,
         window='hann',
         center=True):
    """Short Time Fourier Transform.

    Currently a wrapper for librosa.stft. For full documentation on each
    parameter, checkout https://librosa.org/doc/main/generated/librosa.stft.html

    Parameters
    ----------
    signal : numpy.array
        An array containing audio signal.
    n_fft : int
        Length of the windowed signal after padding with zeros.
    hop_length : int
        Number of audio samples between adjacent STFT columns.
    win_length : int
        Each frame of audio is windowed by window of length win_length and then
        padded with zeros to match n_fft.
    window : str, numpy.array, callable
        A window specification by name, array (of size n_fft) or function.
        Defaults to a raised cosine window (‘hann’), which is adequate for most
        applications in audio signal processing
    center : bool
        If True, the signal y is padded so that frame D[:, t] is centered at
        signal[t * hop_length]. If False, then D[:, t] begins at
        signal[t * hop_length].

    Returns
    -------
    numpy.array
        Short Time Fourier Transform of input signal.

    """
    return librosa.stft(y=signal,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window,
                        center=center)


def spectrogram(signal,
                n_fft,
                hop_length):
    """Create standard spectrogram.

    The absolute value of the STFT of the signal. STFT computation is currently
    accomplished using librosa.stft.

    Parameters
    ----------
    signal : numpy.array
        An array containing audio signal.
    n_fft : int
        Length of the windowed signal after padding with zeros.
    hop_length : int
        Number of audio samples between adjacent STFT columns.

    Returns
    -------
    numpy.array
        Absolute value of STFT.

    """
    return np.abs(stft(signal,
                       n_fft,
                       hop_length))


def spec_frequencies(samplerate,
                     n_fft):
    """Frequency vector for stft parameters.

    Parameters
    ----------
    samplerate : int
        Corresponding samplerate of a signal.
    n_fft : int
        Length of the windowed signal after padding with zeros.

    Returns
    -------
    numpy.array
        Array containing frequency values for vertical bins on a STFT computed
        with given 'n_fft' and a signal having the specified samplerate.

    """
    return librosa.core.fft_frequencies(sr=samplerate,
                                        n_fft=n_fft)
