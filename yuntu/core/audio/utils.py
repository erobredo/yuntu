
import os
import hashlib
import wave
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def binaryMD5(path):
    if path is not None:
        if os.path.isfile(path):
            BLOCKSIZE = 65536
            hasher = hashlib.md5()
            with open(path,"rb") as media:
                buf = media.read(BLOCKSIZE)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = media.read(BLOCKSIZE)
            return hasher.hexdigest()
        else:
            return None
    else:
        return None

def shannon(v):
    sum_v = np.sum(v)
    s = None
    if sum_v > 0:
        p = v/sum_v
        s = -(np.sum(p*np.log(p)))

    return s

def media_size(path):
    if path is not None:
        if os.path.isfile(path):
            return os.path.getsize(path)
        else:
            return None
    else:
        return None

def read_info(path):
    wav = wave.open(path)
    return wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes(), media_size(path)

    
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

def stft(sig,n_fft,hop_length,win_length=None, window='hann', center=True, pad_mode='reflect'):
    return librosa.stft(sig,n_fft,hop_length,win_length=win_length, window=window, center=center, pad_mode=pad_mode)

def spectrogram(sig,n_fft,hop_length):
    return np.abs(stft(sig,n_fft,hop_length))

def spec_frequencies(sr,n_fft):
    return librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

def zero_crossing_rate(sig, frame_length=2048, hop_length=512,center=True):
    return librosa.feature.zero_crossing_rate(sig,frame_length,hop_length,center=center)

def mfcc(sig,sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho'):
    return librosa.feature.mfcc(sig,sr,S, n_mfcc, dct_type, norm)


def plot_power_spec(spec,ax,sr):
    return librosa.display.specshow(librosa.amplitude_to_db(spec,ref=np.max),ax=ax,y_axis='linear',x_axis='time',sr=sr)

def plot_waveform(sig,sr,ax,wtype="simple"):
    if wtype != "simple":
        y_harm, y_perc = librosa.effects.hpss(sig)
        librosa.display.waveplot(y_harm, ax=ax, sr=sr, alpha=0.25)
        return librosa.display.waveplot(y_perc, ax=ax, sr=sr, color='r', alpha=0.5)
    else:
        return librosa.display.waveplot(sig,sr=sr,ax=ax)










