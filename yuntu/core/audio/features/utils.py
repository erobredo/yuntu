"""Tools for audio transformation."""
import numpy as np
import librosa


def stft(signal,
         n_fft,
         hop_length,
         win_length=None,
         window='hann',
         center=True):
    """Short Time Fourier Transform."""
    return librosa.stft(signal,
                        n_fft,
                        hop_length,
                        win_length,
                        window,
                        center)


def spectrogram(signal,
                n_fft,
                hop_length):
    """Create standard spectrogram."""
    return np.abs(stft(signal,
                       n_fft,
                       hop_length))


def spec_frequencies(samplerate,
                     n_fft):
    """Frequency vector for stft parameters."""
    return librosa.core.fft_frequencies(sr=samplerate,
                                        n_fft=n_fft)
