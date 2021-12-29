"""Dataframe accesors for soundscape methods"""
import numpy as np
import pandas as pd
import datetime
import pytz
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from yuntu.soundscape.utils import absolute_timing, aware_time
from yuntu.soundscape.hashers.base import Hasher
from yuntu.soundscape.hashers.crono import CronoHasher, DEFAULT_HASHER_CONFIG
from yuntu.soundscape.pipelines.build_soundscape import HashSoundscape, AbsoluteTimeSoundscape

ID = 'id'
START_TIME = 'start_time'
END_TIME = 'end_time'
MAX_FREQ = 'max_freq'
MIN_FREQ = 'min_freq'
WEIGHT = 'weight'

REQUIRED_SOUNDSCAPE_COLUMNS = [ID,
                               START_TIME,
                               END_TIME,
                               MAX_FREQ,
                               MIN_FREQ]
TIME = "time_raw"
TIME_FORMAT = "time_format"
TIME_ZONE = "time_zone"

CRONO_SOUNDSCAPE_COLUMNS = [TIME, TIME_FORMAT, TIME_ZONE]

@pd.api.extensions.register_dataframe_accessor("sndscape")
class SoundscapeAccessor:
    def __init__(self, pandas_obj):
        id_column = ID
        start_time_column = START_TIME
        end_time_column = END_TIME
        max_freq_column = MAX_FREQ
        min_freq_column = MIN_FREQ
        weight_column = WEIGHT
        time_column = TIME
        time_format_column = TIME_FORMAT
        time_zone_column = TIME_ZONE

        self.index_columns = None
        self._is_crono = None
        self._validate(self, pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(self, obj):
        for col in REQUIRED_SOUNDSCAPE_COLUMNS:
            if col not in obj.columns:
                message = f"Not a soundscape. Missing '{col}' column."
                raise ValueError(message)

        self.index_columns = []
        for col, dtype in zip(list(obj.columns), list(obj.dtypes)):
            if pd.api.types.is_float_dtype(dtype) and col not in REQUIRED_SOUNDSCAPE_COLUMNS:
                self.index_columns.append(col)
        self.index_columns = list(set(self.index_columns))

        if len(self.index_columns) == 0:
            message = "Could not find any column to treat as an acoustic index."
            raise ValueError(message)

        self._is_crono = True
        for col in CRONO_SOUNDSCAPE_COLUMNS:
            if col not in obj.columns:
                self._is_crono = False

    def add_absolute_time(self):
        """Add absolute reference from UTC time"""
        print("Generating absolute time reference...")
        out = self._obj[list(self._obj.columns)]
        out["abs_start_time"] = self._obj.apply(lambda x: absolute_timing(x["time_utc"], x["start_time"]), axis=1)
        out["abs_end_time"] = self._obj.apply(lambda x: absolute_timing(x["time_utc"], x["end_time"]), axis=1)
        return out

    def apply_absolute_time(self, name="apply_absolute_time", work_dir="/tmp", persist=True,
                            read=False, npartitions=1, client=None, show_progress=True,
                            compute=True, time_col="start_time", time_utc_column="abs_start_time", **kwargs):
        """Add absolute reference from UTC time"""
        print("Generating absolute time reference...")
        pipeline = AbsoluteTimeSoundscape(name=name,
                                          work_dir=work_dir,
                                          soundscape_pd=self._obj,
                                          time_col=time_col,
                                          time_utc_column=time_utc_column,
                                          **kwargs)
        if read:
            tpath = os.path.join(work_dir, name, "persist", "absolute_timed_soundscape.parquet")
            if not os.path.exists(tpath):
                raise ValueError(f"Cannot read soundscape. Target file {tpath} does not exist.")
            print("Reading soundscape from file...")
            return pipeline["absolute_timed_soundscape"].read().compute()

        pipeline["absolute_timed_soundscape"].persist = persist
        if compute:
            print("Computing soundscape...")
            if show_progress:
                with ProgressBar():
                    df = pipeline["absolute_timed_soundscape"].compute(client=client,
                                                                       feed={"npartitions": npartitions})
            else:
                df = pipeline["absolute_timed_soundscape"].compute(client=client,
                                                                   feed={"npartitions": npartitions})

            return df
        return pipeline["absolute_timed_soundscape"].future(client=client, feed={"npartitions": npartitions})

    def apply_hash(self, name="apply_hash", work_dir="/tmp", persist=True,
                   read=False, npartitions=1, client=None, show_progress=True,
                   compute=True, **kwargs):
        """Apply indices and produce soundscape."""
        print("Hashing dataframe...")
        pipeline = HashSoundscape(name=name,
                                  work_dir=work_dir,
                                  soundscape_pd=self._obj,
                                  **kwargs)
        if read:
            tpath = os.path.join(work_dir, name, "persist", "hashed_soundscape.parquet")
            if not os.path.exists(tpath):
                raise ValueError(f"Cannot read soundscape. Target file {tpath} does not exist.")
            print("Reading soundscape from file...")
            return pipeline["hashed_soundscape"].read().compute()

        pipeline["hashed_soundscape"].persist = persist
        if compute:
            print("Computing soundscape...")
            if show_progress:
                with ProgressBar():
                    df = pipeline["hashed_soundscape"].compute(client=client,
                                                               feed={"npartitions": npartitions})
            else:
                df = pipeline["hashed_soundscape"].compute(client=client,
                                                           feed={"npartitions": npartitions})

            return df
        return pipeline["hashed_soundscape"].future(client=client, feed={"npartitions": npartitions})

    def add_hash(self, hasher, out_name="xhash"):
        """Add row hasher"""
        print("Hashing dataframe...")
        if not hasher.validate(self._obj):
            str_cols = str(hasher.columns)
            message = ("Input dataframe is incompatible with hasher."
                       f"Missing column inputs. Hasher needs: {str_cols} ")
            raise ValueError(message)

        result = self._obj.apply(hasher, out_name=out_name, axis=1)
        if out_name in list(self._obj.columns):
            raise ValueError(f"Name '{out_name}' not available." +
                             "A column already has that name.")

        out = self._obj[list(self._obj.columns)]
        out[out_name] = result[out_name]

        return out

    def plot_sequence(self, rgb, units=None, view_time_zone="America/Mexico_city", xticks=10,
                      yticks=10, ylabel="Frequency", xlabel="Time", interpolation="bilinear",
                      time_format='%d-%m-%Y %H:%M:%S', ax=None, **kwargs):
        """Plot sequential soundscape."""

        if "abs_start_time" not in self._obj.columns or "abs_end_time" not in self._obj.columns:
            df = self.add_absolute_time()
        else:
            df = self._obj

        utc_zone = pytz.timezone("UTC")
        local_zone = pytz.timezone(view_time_zone)
        if ax is None:
            fig, ax = plt.subplots()

        nfreqs = df["max_freq"].unique().size
        nfeatures = len(rgb)
        min_t = df["abs_start_time"].min()
        max_t = df["abs_end_time"].max()
        max_f = df["max_freq"].max()
        min_f = df["min_freq"].min()

        max_indices = np.expand_dims(np.expand_dims(df[rgb].max().values, axis=0), axis=0)
        min_indices = np.expand_dims(np.expand_dims(df[rgb].min().values, axis=0), axis=0)
        ranges = max_indices - min_indices

        if units is not None:
            time_unit, freq_unit = units
            total_time = datetime.timedelta.total_seconds(max_t - min_t)
            nframes = int(np.round(total_time/time_unit))
            count_matrix = np.zeros([nfreqs, nframes, 1])
            norm_feature_spec = np.zeros([nfreqs, nframes, len(rgb)])

            for val in df[["abs_start_time", "abs_end_time", "max_freq", "min_freq"]+rgb].values:
                abs_start_time, abs_end_time, max_freq, min_freq = val[0:4]
                indices = np.reshape(np.array(val[4:]), [1,1,nfeatures])
                start = int(np.round(float(datetime.timedelta.total_seconds(abs_start_time - min_t))/time_unit))
                stop = int(np.round(float(datetime.timedelta.total_seconds(abs_end_time - min_t))/time_unit))
                maxf = int(np.round((max_freq - min_f)/freq_unit))
                minf = int(np.round((min_freq - min_f)/freq_unit))
                count_matrix[minf:maxf, start:stop+1] += 1
                norm_feature_spec[minf:maxf, start:stop+1, :] = (norm_feature_spec[minf:maxf, start:stop+1,:]
                                                                 + ((indices-min_indices)/ranges))

            norm_feature_spec = np.flip(norm_feature_spec / np.where(count_matrix > 0, count_matrix, 1), axis=0)

        else:
            snd_matrix = np.flip(np.reshape(df.sort_values(by=["max_freq", "abs_start_time"])[rgb].values, [nfreqs,-1,len(rgb)]),axis=0)
            norm_feature_spec = (snd_matrix - min_indices) / ranges

        ntimes = norm_feature_spec.shape[1]

        if norm_feature_spec.shape[-1] == 2:
            norm_feature_spec = np.concatenate([norm_feature_spec,
                                                np.zeros([norm_feature_spec.shape[0],
                                                          norm_feature_spec.shape[1], 1])], axis=-1)

        im = ax.imshow(np.flip(norm_feature_spec, axis=0), aspect="auto", interpolation=interpolation, **kwargs)

        tstep = float(ntimes)/xticks
        ax.set_xticks(np.arange(0,ntimes+tstep,tstep))
        tlabel_step = (max_t - min_t) / xticks

        try:
            tlabels = [utc_zone.localize(min_t+tlabel_step*i) for i in range(xticks)]+[utc_zone.localize(max_t)]
        except:
            tlabels = [(min_t+tlabel_step*i) for i in range(xticks)]+[max_t]

        ax.set_xticklabels([t.astimezone(local_zone).strftime(format=time_format) for t in tlabels])
        ax.invert_yaxis()

        fstep = float(nfreqs)/yticks
        ax.set_yticks(np.arange(0,nfreqs+fstep,fstep))
        flabel_step = float(max_f-min_f)/yticks
        yticklabels = ["{:.2f}".format(x) for x in list(np.arange(min_f/1000,(max_f+flabel_step)/1000,flabel_step/1000))]
        ax.set_yticklabels(yticklabels)

        ax.set_ylabel(f"{ylabel} (kHz)")
        ax.set_xlabel(f"{xlabel} ({time_format}, {view_time_zone})")

        return ax, im

    def plot_cycle(self, rgb, hash_col=None, cycle_config=DEFAULT_HASHER_CONFIG, aggr="mean", xticks=10,
                   yticks=10, ylabel="Frequency", xlabel="Time", interpolation="bilinear",
                   time_format='%H:%M:%S', view_time_zone="America/Mexico_city", ax=None, **kwargs):
        """Plot soundscape according to cycle configs."""
        if ax is None:
            ax = plt.gca()

        time_module = cycle_config["time_module"]
        time_unit = cycle_config["time_unit"]
        zero_t = None

        if "aware_start" in cycle_config:
            if cycle_config["aware_start"] is not None:
                zero_t = cycle_config["aware_start"]
        if zero_t is None:
            if "start_time" in cycle_config and "start_tzone" in cycle_config and "start_format" in cycle_config:
                zero_t = aware_time(cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"])
            else:
                raise ValueError("Must provide starting time information to interpret hash")
        local_zone = pytz.timezone(view_time_zone)
        do_hash = False
        hash_name = "crono_hasher"
        if hash_col is None:
            do_hash = True
        elif hash_col not in list(self._obj.columns):
            hash_name = hash_col
            do_hash = True
        else:
            hash_name = hash_col

        df = self._obj
        if do_hash:
            hasher = CronoHasher(**cycle_config)
            hashed_df = df.sndscape.add_hash(hasher, out_name=hash_name)
        else:
            hashed_df = df

        max_hash = time_module
        all_hashes = list(np.arange(0, max_hash))

        missing_hashes = [x for x in all_hashes if x not in list(hashed_df[hash_name].unique())]
        nfreqs = hashed_df["max_freq"].unique().size

        max_f = hashed_df["max_freq"].max()
        min_f = hashed_df["min_freq"].min()
        nfeatures = len(rgb)

        max_indices = hashed_df[rgb].max().values
        min_indices = hashed_df[rgb].min().values
        ranges = max_indices - min_indices

        proj_df = hashed_df[["max_freq", hash_name]+rgb].copy()

        for n, ind in enumerate(rgb):
            proj_df.loc[:, ind] = proj_df[ind].apply(lambda x: (x - min_indices[n]) / ranges[n])

        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        min_t = proj_df[f"{hash_name}_time"].min()
        max_t = proj_df[f"{hash_name}_time"].max()

        proj_df = proj_df[["max_freq", f"{hash_name}_time"]+rgb]
        norm_feature_spec = (np.flip(np.reshape(proj_df
                                                .groupby(by=["max_freq", f"{hash_name}_time"], as_index=True)
                                                .agg(aggr)
                                                .reset_index()
                                                .sort_values(by=["max_freq", f"{hash_name}_time"])[rgb].values, [nfreqs,-1,nfeatures]),axis=0))

        ntimes = norm_feature_spec.shape[1]

        # Fill missing units
        null_arr = np.empty([nfreqs,nfeatures])
        null_arr[:,:] = np.NaN
        for x in missing_hashes:
            norm_feature_spec = np.insert(norm_feature_spec, x+1, null_arr, 1)

        if norm_feature_spec.shape[-1] == 2:
            norm_feature_spec = np.concatenate([norm_feature_spec,
                                                np.zeros([norm_feature_spec.shape[0],
                                                          norm_feature_spec.shape[1], 1])], axis=-1)
        im = ax.imshow(np.flip(norm_feature_spec, axis=0), aspect="auto", interpolation=interpolation, **kwargs)
        tstep = float(time_module)/xticks
        ax.set_xticks(np.arange(0,time_module,tstep))
        tlabel_step = datetime.timedelta(seconds=time_unit)*time_module / xticks

        ax.set_xticklabels([(min_t+tlabel_step*i).astimezone(local_zone).strftime(format=time_format)
                           for i in range(xticks)])

        ax.invert_yaxis()
        fstep = float(nfreqs)/yticks
        ax.set_yticks(np.arange(0,nfreqs+fstep,fstep))
        flabel_step = float(max_f-min_f)/yticks
        yticklabels = ["{:.2f}".format(x) for x in list(np.arange(min_f/1000,(max_f+flabel_step)/1000,flabel_step/1000))]
        ax.set_yticklabels(yticklabels)

        ax.set_ylabel(f"{ylabel} (kHz)")
        ax.set_xlabel(f"{xlabel} ({time_format}, {view_time_zone})")

        return ax, im
