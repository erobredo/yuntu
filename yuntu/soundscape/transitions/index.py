"""Transitions for acoustic indices."""
import os
import numpy as np
import pandas as pd
import datetime

from yuntu.core.pipeline.transitions.decorators import transition
from yuntu.core.pipeline.places import PickleablePlace
from yuntu.core.pipeline.places.extended import DaskDataFramePlace
from yuntu.core.pipeline.places import *
from yuntu.soundscape.utils import slice_windows, sliding_slice_windows, aware_time

def feature_indices(row, indices):
    """Compute acoustic indices for one row."""
    new_row = {}
    for index in indices:
        new_row[index.name] = index(row['feature_cut'])
    return pd.Series(new_row)


def feature_slices(row, audio, config, indices):
    """Produce slices from recording and configuration."""
    cuts, weights = slice_windows(config["time_unit"],
                                  audio.duration,
                                  config["frequency_bins"],
                                  config["frequency_limits"],
                                  config["time_hop"],
                                  config["frequency_hop"])
    feature = getattr(audio.features,
                      config["feature_type"])(**config["feature_config"])
    audio.clean()
    feature_cuts = [feature.cut_array(cut) for cut in cuts]
    feature.clean()

    start_times = [cut.start for cut in cuts]
    end_times = [cut.end for cut in cuts]
    max_freqs = [cut.max for cut in cuts]
    min_freqs = [cut.min for cut in cuts]

    new_row = {}
    new_row['start_time'] = start_times
    new_row['end_time'] = end_times
    new_row['min_freq'] = max_freqs
    new_row['max_freq'] = min_freqs
    new_row['weight'] = weights

    for index in indices:
        results = []
        if index.ncomponents > 1:
            base_name = index.name
            cut_results = np.concatenate([index(fcut) for fcut in feature_cuts], axis=0)
            for n in range(index.ncomponents):
                subindex_name = f'{base_name}_{n}'
                new_row[subindex_name] = cut_results[:,n]
        else:
            for fcut in feature_cuts:
                results.append(index(fcut))
            new_row[index.name] = results

    return pd.Series(new_row)

def write_timed_grid_slices(row, audio, slice_config, write_config, indices):
    """Produce slices from recording and configuration."""
    cuts, weights = sliding_slice_windows(audio.duration,
                                          slice_config["time_unit"],
                                          slice_config["time_hop"],
                                          slice_config["frequency_limits"],
                                          slice_config["frequency_unit"],
                                          slice_config["frequency_hop"])
    feature = getattr(audio.features,
                      slice_config["feature_type"])(**slice_config["feature_config"])
    audio.clean()
    feature_cuts = [feature.cut_array(cut) for cut in cuts]
    feature.clean()

    strtime = row["time_raw"]
    time_zone = row["time_zone"]
    time_format = row["time_format"]

    atime = aware_time(strtime, time_zone, time_format)

    include_meta = {}
    if "include_meta" in slice_config:
        include_meta = {key: np.array([str(row["metadata"][key])])
                        for key in slice_config["include_meta"]}

    classes = np.load(slice_config["soundscape_classes"])

    columns = ["recording_id", "npz_path",
               "time_class", "frequency_class",
               "soundscape_class",
               "start_time", "end_time",
               "min_freq", "max_freq",
               "time_raw", "time_format",
               "time_zone"]

    for index in indices:
        columns.append(index.name)

    packed_results = {key:[] for key in columns}
    recording_id = row["id"]
    recording_path = row["path"]
    time_class = row["time_class"]
    columns = []
    basename, _ = os.path.splitext(os.path.basename(recording_path))
    for n, cut in enumerate(cuts):
        start_time = cut.start
        min_freq = cut.min
        end_time = cut.end
        max_freq = cut.max
        bounds = [start_time, min_freq, end_time, max_freq]
        frequency_class = int(min_freq/1000)

        c1 = int(classes[[time_class], [frequency_class], [0]])
        c2 = int(classes[[time_class], [frequency_class], [1]])
        c3 = int(classes[[time_class], [frequency_class], [2]])
        soundscape_class = f"{c1}{c2}{c3}"

        start_datetime = atime + datetime.timedelta(seconds=start_time)
        piece_time_raw = start_datetime.strftime(format=time_format)
        chunck_basename = '%.2f_%.2f_%.2f_%.2f' % tuple(bounds)
        chunck_file = f'{basename}_{chunck_basename}_{recording_id}.npz'

        npz_path = os.path.join(write_config["write_dir"], soundscape_class, chunck_file)
        output = {
             "bounds": np.array(bounds),
             "recording_path": np.array([recording_path]),
             "array": feature_cuts[n]
        }

        index_results = {}
        for index in indices:
            index_result = index(feature_cuts[n])
            index_results[index.name] = index_result
            output[index.name] = np.array([index_result])

        output.update(include_meta)

        np.savez_compressed(npz_path, **output)

        new_row = {}
        new_row["recording_id"] = recording_id
        new_row["npz_path"] = npz_path
        new_row["time_class"] = time_class
        new_row["frequency_class"] = frequency_class
        new_row["soundscape_class"] = soundscape_class
        new_row['start_time'] = start_time
        new_row['end_time'] = end_time
        new_row['min_freq'] = max_freq
        new_row['max_freq'] = min_freq
        new_row['time_raw'] = piece_time_raw
        new_row['time_format'] = time_format
        new_row['time_zone'] = time_zone

        new_row.update(index_results)

        for c in columns:
            packed_results[c].append(new_row[c])

    return pd.Series(packed_results)


@transition(name='slice_features', outputs=["feature_slices"], persist=True,
            signature=((DaskDataFramePlace, DictPlace, PickleablePlace),
                       (DaskDataFramePlace,)))
def slice_features(recordings, config, indices):
    """Produce feature slices dataframe."""

    meta = [('start_time', np.dtype('float64')),
            ('end_time', np.dtype('float64')),
            ('min_freq', np.dtype('float64')),
            ('max_freq', np.dtype('float64')),
            ('weight', np.dtype('float64'))]

    single_indices = [index for index in indices if index.ncomponents == 1]
    multi_indices = [index for index in indices if index.ncomponents > 1]

    meta += [(index.name,
             np.dtype('float64'))
             for index in single_indices]

    for index in multi_indices:
        base_name = index.name
        for n in range(index.ncomponents):
            subindex_name = f'{base_name}_{n}'
            meta.append((subindex_name, np.dtype('float64')))

    result = recordings.audio.apply(feature_slices,
                                    meta=meta,
                                    config=config,
                                    indices=indices)
    dropcols = ['datastore', 'path', 'hash', 'timeexp', 'spectrum',
                'classtype', 'duration', 'filesize', 'length', 'nchannels',
                'samplerate', 'sampwidth', 'metadata']

    colnames = [name for name in list(recordings.columns) if name not in dropcols]
    subrecs = recordings[colnames]

    subrecs['start_time'] = result['start_time']
    slices = subrecs.explode('start_time')
    slices['end_time'] = result['end_time'].explode()
    slices['min_freq'] = result['max_freq'].explode()
    slices['max_freq'] = result['min_freq'].explode()
    slices['weight'] = result['weight'].explode()

    types = {"start_time": "float64",
             "end_time": "float64",
             "min_freq": "float64",
             "max_freq": "float64",
             "weight": "float64"}

    for index in single_indices:
        slices[index.name] = result[index.name].explode()
        types[index.name] = "float64"

    for index in multi_indices:
        base_name = index.name
        for n in range(index.ncomponents):
            subindex_name = f'{base_name}_{n}'
            slices[subindex_name] = result[subindex_name].explode()
            types[subindex_name] = "float64"

    return slices.astype(types)

@transition(name='apply_indices', outputs=["index_results"],
            is_output=True, persist=True, keep=True,
            signature=((DaskDataFramePlace, PickleablePlace),
                       (DaskDataFramePlace, )))
def apply_indices(slices, indices):
    """Apply acoustic indices to slices."""
    index_names = [index.name for index in indices]
    if len(index_names) != len(set(index_names)):
        message = "Index names have duplicates. Please use a diferent name" + \
                  " for each index to compute."
        raise ValueError(message)

    meta = [(index.name,
            np.dtype('float64'))
            for index in indices]

    results = slices.apply(feature_indices,
                           meta=meta,
                           axis=1,
                           indices=indices)
    for index in indices:
        slices[index.name] = results[index.name]

    return slices.drop(['feature_cut'], axis=1)


@transition(name='slice_samples', outputs=["slice_results"], persist=True,
            signature=((DaskDataFramePlace, DictPlace, DictPlace, DynamicPlace),
                       (DaskDataFramePlace,)))
def slice_timed_samples(hashed_dd, slice_config, write_config, indices):
    """Produce feature slices dataframe."""
    if not os.path.exists(write_config["write_dir"]):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    os.makedirs(os.path.join(write_config["write_dir"], f"{i}{j}{k}"))

    meta = [('recording_id', np.dtype(int)),
            ('npz_path', np.dtype('<U')),
            ('time_class', np.dtype(int)),
            ('frequency_class', np.dtype(int)),
            ('soundscape_class', np.dtype('<U')),
            ('start_time', np.dtype('float64')),
            ('end_time', np.dtype('float64')),
            ('min_freq', np.dtype('float64')),
            ('max_freq', np.dtype('float64')),
            ('time_raw', np.dtype('<U')),
            ('time_format', np.dtype('<U')),
            ('time_zone', np.dtype('<U'))]

    meta = meta + [(index.name, np.dtype('float64')) for index in indices]

    results = hashed_dd.audio.apply(write_timed_grid_slices,
                                    meta=meta,
                                    slice_config=slice_config,
                                    write_config=write_config,
                                    indices=indices)
    hdd = hashed_dd[["path"]]
    hdd["recording_id"] = results["recording_id"]
    slices = hdd.explode('recording_id')

    for obj in meta:
        if obj[0] != "recording_id":
            slices[obj[0]] = results[obj[0]].explode()

    return slices
