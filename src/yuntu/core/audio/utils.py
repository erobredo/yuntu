"""Auxiliar utilities for Audio classes and methods."""
from typing import Optional
import os
import glob
import io
import hashlib
import wave
import numpy as np
import librosa
import soundfile
import shutil

SAMPWIDTHS = {
    'PCM_16': 2,
    'PCM_32': 4,
    'PCM_U8': 1,
    'FLOAT': 4
}

def s3_glob(path):
    '''Glob using s3fs'''
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    return s3.glob(path)

def ag_glob(path):
    '''Agnostic glob'''
    if path[:5] == "s3://":
        return s3_glob(path)
    return glob.glob(path)

def media_open_s3(path):
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    bucket = path.replace("s3://", "").split("/")[0]
    key = path.replace(f"s3://{bucket}/", "")
    return s3.open('{}/{}'.format(bucket, key))

def media_copy_s3(source_path, target_path):
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    if source_path[:5] == "s3://" and target_path[:5] == "s3://":
        return s3.copy(source_path, target_path)
    elif source_path[:5] == "s3://":
        return s3.download(source_path, target_path)
    return s3.upload(source_path, target_path)

def media_copy(source_path, target_path):
    if source_path[:5] == "s3://" or target_path[:5] == "s3://":
        return media_copy_s3(source_path, target_path)
    return shutil.copy(source_path, target_path)

def media_open(path, mode='rb'):
    if path[:5] == "s3://":
        return media_open_s3(path)
    return open(path, mode)

def media_exists(path):
    if path[:5] == "s3://":
        from s3fs.core import S3FileSystem
        s3 = S3FileSystem()
        return s3.exists(path)
    return os.path.exists(path)

def load_media_info_s3(path):
    """Return media size of s3 locations."""
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    info = s3.info(path)
    if "size" in info:
        if info["size"] is not None:
            filesize = info["size"]
    elif "Size" in info:
        if info["Size"] is not None:
            filesize = info["Size"]
    else:
        raise ValueError(f"Could not retrieve size of file {path}")

    audio_info = soundfile.info(media_open_s3(path))

    return audio_info.samplerate, audio_info.channels, audio_info.duration, audio_info.subtype, filesize

def load_media_info(path):
    if path[:5] == "s3://":
        return load_media_info_s3(path)
    audio_info = soundfile.info(path)
    filesize = media_size(path)
    return audio_info.samplerate, audio_info.channels, audio_info.duration, audio_info.subtype, filesize

def binary_md5(path, blocksize=65536):
    """Hash file by blocksize."""
    if path is None:
        raise ValueError("Path is None.")
    if not media_exists(path):
        raise ValueError("Path does not exist.")

    hasher = hashlib.md5()
    with media_open(path, "rb") as media:
        buf = media.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = media.read(blocksize)
    return hasher.hexdigest()

def media_size(path):
    """Return media size or None."""
    if isinstance(path, io.BytesIO):
        return len(path.getvalue())

    if path is not None:
        if os.path.isfile(path):
            return os.path.getsize(path)
    return None

def read_info(path, timeexp):
    """Read recording information form file."""
    samplerate, nchannels, duration, subtype, filesize = load_media_info(path)
    media_info = {}
    media_info["samplerate"] = samplerate
    media_info["nchannels"] = nchannels
    media_info["sampwidth"] = SAMPWIDTHS[subtype]
    media_info["length"] = int(duration*samplerate)
    media_info["filesize"] = filesize
    media_info["duration"] = float(duration) / timeexp
    return media_info

def hash_file(path, alg="md5"):
    """Produce hash from audio recording."""
    if alg == "md5":
        return binary_md5(path)
    raise NotImplementedError("Algorithm "+alg+" is not implemented.")

def read_media(path,
               samplerate,
               offset=0.0,
               duration=None,
               **kwargs):
    """Read media."""
    if path[:5] == "s3://":
        path = media_open_s3(path)
    return librosa.load(path,
                        sr=samplerate,
                        offset=offset,
                        duration=duration,
                        mono=True,
                        **kwargs)

def write_media(path,
                signal,
                samplerate,
                nchannels,
                media_format="wav"):
    """Write media."""
    if media_format not in ["wav", "flac", "ogg"]:
        raise NotImplementedError("Writer for " + media_format
                                  + " not implemented.")
    if nchannels > 1:
        signal = np.transpose(signal, (1, 0))
    soundfile.write(path,
                    signal,
                    samplerate,
                    format=media_format)


def get_channel(signal, channel, nchannels):
    """Return correct channel in any case."""
    if nchannels > 1:
        return np.squeeze(signal[[channel], :])
    return signal


def resample(
        array: np.array,
        original_sr: int,
        target_sr: int,
        res_type: Optional[str] = 'kaiser_best',
        fix: Optional[bool] = True,
        scale: Optional[bool] = False,
        **kwargs):
    """Resample audio."""
    return librosa.core.resample(
        array,
        original_sr,
        target_sr,
        res_type=res_type,
        fix=fix,
        scale=scale,
        **kwargs)


def channel_mean(signal, keepdims=False):
    """Return channel mean."""
    return np.mean(signal,
                   axis=0,
                   keepdims=keepdims)
