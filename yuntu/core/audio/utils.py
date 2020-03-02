import os
import wave
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def mediaSize(path):
    if path is not None:
        if os.path.isfile(path):
            return os.path.getsize(path)
        else:
            return None
    else:
        return None

def readInfo(path):
    wav = wave.open(path)
    return wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes(), mediaSize(path)


def read(path,sr,offset=0.0,duration=None):
    return librosa.load(path,sr=sr,offset=offset,duration=duration,mono=False)

def write(path,sig,sr,nchannels,media_format="wav"):
    if nchannels > 1:
        sig = np.transpose(sig,(1,0))
    sf.write(path,sig,sr,format=media_format)

def sigChannel(sig,channel,nchannels):
    if nchannels > 1:
        return np.squeeze(sig[[channel],:])
    else:
        return sig

def channelMean(sig,keepdims=False):
    return np.mean(sig,axis=0,keepdims=keepdims)

def stft(sig,n_fft,hop_length,win_length=None, window='hann', center=True, pad_mode='reflect'):
    return librosa.stft(sig,n_fft,hop_length,win_length=win_length, window=window, center=center, pad_mode=pad_mode)

def spectrogram(sig,n_fft,hop_length):
    return np.abs(stft(sig,n_fft,hop_length))

def specFrequencies(sr,n_fft):
    return librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

def zeroCrossingRate(sig, frame_length=2048, hop_length=512,center=True):
    return librosa.feature.zero_crossing_rate(sig,frame_length,hop_length,center=center)

def mfcc(sig,sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho'):
    return librosa.feature.mfcc(sig,sr,S, n_mfcc, dct_type, norm)

def minMaxNorm(x):
    amin = np.amin(x)
    amax = np.amax(x)

    return (x - amin)/(amax - amin)

