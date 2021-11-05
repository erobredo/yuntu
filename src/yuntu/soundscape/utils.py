"""Utilities for soundscape operations and indices."""
import itertools
import json
import numpy as np
from yuntu.core.windows import TimeFrequencyWindow
import datetime,time
import pytz
from pytz import timezone


def interpercentile_power_mean(power, ref, perc_ranges):
    """Return mean power in percentile ranges."""
    perc_ranges = list(set(perc_ranges))
    nranges = len(perc_ranges)
    arr_filter = ((ref > np.percentile(ref, perc_ranges[0][0])) &
                  (ref <= np.percentile(ref, perc_ranges[0][1])))
    if nranges > 1:
        for i in range(1, nranges):
            arr_filter = arr_filter | \
                         ((ref > np.percentile(ref, perc_ranges[i][0])) &
                          (ref <= np.percentile(ref, perc_ranges[i][1])))
    return np.mean(power[arr_filter])


def interpercentile_mean_decibels(power, ref, perc_ranges):
    """Return decibels of mean power in percentile ranges."""
    return 10*np.log10(interpercentile_power_mean(power, ref, perc_ranges))


def decile_mod(x, tolerance=0.1):
    """Return percentile ranges of modes."""
    hist, bin_edges = np.histogram(x, bins=100)
    max_count = np.max(hist)
    perc_arr = np.array(range(0, 101))
    percentiles = np.percentile(x, perc_arr)
    mods = []
    mod_deciles = []
    perc_ranges = []
    for i in range(len(hist)):
        if hist[i] >= max_count-tolerance*max_count:
            mod = (bin_edges[i+1]+bin_edges[i])/2
            mods.append(mod)

    for mod in mods:
        mod_diff = np.abs(percentiles-mod)
        min_diff = np.min(mod_diff)
        mod_percentile = int(np.round(np.mean(perc_arr[mod_diff == min_diff])))
        mod_decile = min(int(mod_percentile/10)+1, 10)
        mod_deciles.append(mod_decile)
        perc_ranges.append(((mod_decile-1)*10, mod_decile*10))

    return mods, mod_deciles, perc_ranges

def slice_windows(time_unit, duration, frequency_bins, frequency_limits,
                  time_hop=1.0, frequency_hop=1.0):
    """Produce a list of time frequency windows."""
    frequency_unit = (frequency_limits[1]-frequency_limits[0]) / frequency_bins
    time_hop = time_unit*time_hop
    frequency_hop = frequency_unit*frequency_hop

    bounds = itertools.product([(t, t+time_unit)
                                for t in np.arange(0, duration, time_hop)
                                if t+time_unit <= duration],
                               [(f, f+frequency_unit)
                                for f in np.arange(frequency_limits[0],
                                                   frequency_limits[1],
                                                   frequency_hop)
                                                   if f+frequency_unit <= frequency_limits[1]])
    windows = []
    weights = []
    for interval_t, interval_f in bounds:
        start_time, end_time = interval_t
        end_time = min(end_time, duration)
        size_T = end_time - start_time
        if size_T >= time_unit / 2:
            min_frequency, max_frequency = interval_f
            weights.append((end_time - start_time) / time_unit)
            windows.append(TimeFrequencyWindow(start=start_time,
                                               end=end_time,
                                               min=min_frequency,
                                               max=max_frequency))
    return windows, weights


def sliding_slice_windows(duration, time_unit, time_hop,
                          frequency_limits, frequency_unit, frequency_hop):
    bounds = itertools.product([(t, t+time_unit)
                                for t in np.arange(0, duration, time_hop)],
                               [(f, f+frequency_unit)
                                for f in np.arange(frequency_limits[0],
                                                   frequency_limits[1],
                                                   frequency_hop)])
    windows = []
    weights = []
    for interval_t, interval_f in bounds:
        start_time, end_time = interval_t
        end_time = min(end_time, duration)
        size_T = end_time - start_time
        if size_T == time_unit:
            min_frequency, max_frequency = interval_f
            weights.append((end_time - start_time) / time_unit)
            windows.append(TimeFrequencyWindow(start=start_time,
                                               end=end_time,
                                               min=min_frequency,
                                               max=max_frequency))
    return windows, weights


def aware_time(strtime, tzone, time_format):
    dtime = datetime.datetime.strptime(strtime, time_format)
    if dtime.tzinfo is None or dtime.tzinfo.utcoffset(dtime) is None:
        tzobj = pytz.timezone(tzone)
        dtime = tzobj.localize(dtime)
    return dtime

def from_timestamp(timestamp, tzone):
    dtime = timestamp.to_pydatetime()
    if dtime.tzinfo is None or dtime.tzinfo.utcoffset(dtime) is None:
        tzobj = pytz.timezone(tzone)
        dtime = tzobj.localize(dtime)
    return dtime

def time_as_seconds(time_obj):
    tzobj = pytz.timezone("UTC")
    globalstart = tzobj.localize(datetime.datetime(1970, 1, 1, 0, 0, 0))

    return (time_obj - globalstart).total_seconds()

def absolute_timing(time_ref, seconds):
    return time_ref + datetime.timedelta(seconds=seconds)

def parse_json(row, columns):
    for col in columns:
        if isinstance(row[col], (str, bytes, bytearray)):
            row[col] = json.loads(row[col])
    return row
