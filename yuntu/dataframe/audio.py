"""Audio dataframe base classes"""
import os
import pandas as pd
from dask import delayed
import dask.dataframe.extensions
from dask.diagnostics import ProgressBar

from yuntu.core.audio.audio import Audio
from yuntu.soundscape.utils import parse_json
from yuntu.soundscape.pipelines.build_soundscape import Soundscape, CronoSoundscape
from yuntu.soundscape.pipelines.probe_dataframe import ProbeDataframe

PATH = 'path'
SAMPLERATE = 'samplerate'
TIME_EXPANSION = 'timeexp'
DURATION = 'duration'
MEDIA_INFO = 'media_info'
METADATA = 'metadata'
ID = 'id'
ANNOTATIONS = 'annotations'
REQUIRED_AUDIO_COLUMNS = [
    PATH,
]
OPTIONAL_AUDIO_COLUMNS = [
    SAMPLERATE,
    TIME_EXPANSION,
    DURATION,
    MEDIA_INFO,
    METADATA,
    ID,
]


@pd.api.extensions.register_dataframe_accessor("audio")
class AudioAccessor:
    path_column = PATH
    samplerate_column = SAMPLERATE
    timeexp_column = TIME_EXPANSION
    duration_column = DURATION
    media_info_column = MEDIA_INFO
    metadata_column = METADATA
    id_column = ID
    annotations_columns = ANNOTATIONS

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not all(column in obj.columns for column in REQUIRED_AUDIO_COLUMNS):
            raise AttributeError("Must have 'path'")

    def _build_audio(
            self,
            row,
            lazy=True,
            path_column=None,
            samplerate_column=None,
            timeexp_column=None,
            duration_column=None,
            media_info_column=None,
            metadata_column=None,
            annotations_columns=None,
            id_column=None):

        if path_column is None:
            path_column = self.path_column

        if samplerate_column is None:
            samplerate_column = self.samplerate_column

        if timeexp_column is None:
            timeexp_column = self.timeexp_column

        if duration_column is None:
            duration_column = self.duration_column

        if media_info_column is None:
            media_info_column = self.media_info_column

        if metadata_column is None:
            metadata_column = self.metadata_column

        if annotations_columns is None:
            annotations_columns = self.annotations_columns

        if id_column is None:
            id_column = self.id_column

        data = {
            PATH: getattr(row, path_column),
            SAMPLERATE: getattr(row, samplerate_column, None),
            TIME_EXPANSION: getattr(row, timeexp_column, None),
            DURATION: getattr(row, duration_column, None),
            MEDIA_INFO: getattr(row, media_info_column, None),
            METADATA: getattr(row, metadata_column, None),
            ID: getattr(row, id_column, None),
            ANNOTATIONS: getattr(row, annotations_columns, [])
        }

        return Audio(**data, lazy=lazy)

    def apply(self, func,**kwargs):
        return self._obj.apply(
            lambda row: func(row, self._build_audio(row), **kwargs),
            axis=1)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._build_audio(self._obj.iloc[key])

        return [
            self._build_audio(row)
            for row in self._obj[key].itertuples()]

    def get(
            self,
            row=None,
            id=None,
            lazy=True,
            path_column=None,
            samplerate_column=None,
            timeexp_column=None,
            duration_column=None,
            media_info_column=None,
            annotations_columns=None,
            metadata_column=None,
            id_column=None):
        if id_column is None:
            id_column = self.id_column

        if row is not None:
            row = self._obj.iloc[row]
        elif id is not None:
            row = self._obj[self._obj[id_column] == id].iloc[0]
        else:
            row = self._obj.iloc[0]

        return self._build_audio(
            row,
            lazy=lazy,
            path_column=path_column,
            samplerate_column=samplerate_column,
            timeexp_column=timeexp_column,
            duration_column=duration_column,
            media_info_column=media_info_column,
            metadata_column=metadata_column,
            annotations_columns=annotations_columns,
            id_column=id_column)

    def change_path_column(self, new_column):
        self.path_column = new_column

    def change_samplerate_column(self, new_column):
        self.samplerate_column = new_column

    def change_timeexp_column(self, new_column):
        self.timeexp_column = new_column

    def change_duration_column(self, new_column):
        self.duration_column = new_column

    def change_media_info_column(self, new_column):
        self.media_info_column = new_column

    def change_metadata_column(self, new_column):
        self.metadata_column = new_column

    def change_annotations_column(self, new_column):
        self.annotations_columns = new_column

    def change_id_column(self, new_column):
        self.id_column = new_column

    def apply_probe(self, probe_config, batch_size=200, name="apply_probe", work_dir="/tmp", persist=True,
                    read=False, npartitions=1, client=None, show_progress=True,
                    compute=True, **kwargs):
        """Apply probe and return matches."""
        pipeline = ProbeDataframe(name=name,
                                  work_dir=work_dir,
                                  recordings=self._obj,
                                  probe_config=probe_config,
                                  batch_size=batch_size,
                                  **kwargs)
        if read:
            tpath = os.path.join(work_dir, name, "persist", "matches.parquet")
            if not os.path.exists(tpath):
                raise ValueError(f"Cannot read matches. Target file {tpath} does not exist.")
            print("Reading matches from file...")
            return (pipeline["matches"]
                    .read()
                    .compute()
                    .apply(lambda row: parse_json(row, ["labels", "metadata"]), axis=1))

        pipeline["matches"].persist = persist
        if compute:
            print("Applying probe...")

            if show_progress:
                with ProgressBar():
                    df = pipeline["matches"].compute(client=client,
                                                     feed={"npartitions": npartitions})
            else:
                df = pipeline["matches"].compute(client=client,
                                                 feed={"npartitions": npartitions})

            return df.apply(lambda row: parse_json(row, ["labels", "metadata"]), axis=1)
        return pipeline["matches"].future(client=client, feed={"npartitions": npartitions})

    def get_soundscape(self, name="get_soundscape", work_dir="/tmp", persist=True,
                       read=False, npartitions=1, client=None, show_progress=True,
                       compute=True, crono=True, **kwargs):
        """Apply indices and produce soundscape."""
        if crono:
            pipeline = CronoSoundscape(name=name,
                                       work_dir=work_dir,
                                       recordings=self._obj,
                                       **kwargs)
            out_place = "hashed_soundscape"
        else:
            pipeline = Soundscape(name=name,
                                  work_dir=work_dir,
                                  recordings=self._obj,
                                  **kwargs)
            out_place = "soundscape"

        if read:
            tpath = os.path.join(work_dir, name, "persist", f"{out_place}.parquet")
            if not os.path.exists(tpath):
                raise ValueError(f"Cannot read soundscape. Target file {tpath} does not exist.")
            print("Reading soundscape from file...")
            return pipeline[out_place].read().compute()

        pipeline[out_place].persist = persist
        if compute:
            print("Computing soundscape...")
            if show_progress:
                with ProgressBar():
                    df = pipeline[out_place].compute(client=client,
                                                     feed={"npartitions": npartitions})
            else:
                df = pipeline[out_place].compute(client=client,
                                                 feed={"npartitions": npartitions})

            return df
        return pipeline[out_place].future(client=client, feed={"npartitions": npartitions})


def dask_wrapper(func):
    name = func.__name__

    def wrapper(self, *args, **kwargs):

        def delayed_func():
            accesor = self._obj.compute().audio
            method = getattr(accesor, name)
            return method(*args, **kwargs)

        return delayed(delayed_func)()
    return wrapper


@dask.dataframe.extensions.register_dataframe_accessor("audio")
class DaskAudioAccesor(AudioAccessor):
    @dask_wrapper
    def __getitem__(self, key):
        super().__getitem__(key)

    def apply(self, func, args=(), meta='__no_default__', **kwargs):
        def wrapper(row, *nargs, **kwargs):
            audio = self._build_audio(row)
            return func(row, audio, *nargs, **kwargs)

        return self._obj.apply(
            wrapper,
            axis=1,
            args=args,
            meta=meta,
            **kwargs)

    # pylint: disable=redefined-builtin, too-many-arguments
    @dask_wrapper
    def get(
            self,
            row=None,
            id=None,
            lazy=True,
            path_column=None,
            samplerate_column=None,
            timeexp_column=None,
            duration_column=None,
            media_info_column=None,
            metadata_column=None,
            id_column=None):
        return super().get(
            row=None,
            id=None,
            lazy=True,
            path_column=None,
            samplerate_column=None,
            timeexp_column=None,
            duration_column=None,
            media_info_column=None,
            metadata_column=None,
            id_column=None)
