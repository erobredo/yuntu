"""Audio dataframe base classes.

An audio dataframe is a
"""
import numpy as np
import pandas as pd
import datetime
from yuntu.core.annotation.annotation import Annotation


GEOMETRY = 'geometry'
TYPE = 'type'
LABELS = 'labels'
ID = 'id'
ABS_START_TIME = "abs_start_time"
ABS_END_TIME = "abs_end_time"

REQUIRED_ANNOTATION_COLUMNS = [
    GEOMETRY,
    TYPE,
]
OPTIONAL_ANNOTATION_COLUMNS = [
    LABELS,
    ABS_START_TIME,
    ABS_END_TIME
]


@pd.api.extensions.register_dataframe_accessor("annotation")
class AnnotationAccessor:
    type_column = TYPE
    geometry_column = GEOMETRY
    labels_column = LABELS
    id_column = ID

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        columns = obj.columns
        for column in REQUIRED_ANNOTATION_COLUMNS:
            if column not in columns:
                raise AttributeError(f"Must have column {column}")

    def _build_annotation(
            self,
            row,
            type_column=None,
            geometry_column=None,
            labels_column=None,
            id_column=None):

        if type_column is None:
            type_column = self.type_column

        if geometry_column is None:
            geometry_column = self.geometry_column

        if labels_column is None:
            labels_column = self.labels_column

        if id_column is None:
            id_column = self.id_column

        data = {
            GEOMETRY: {
                'wkt': getattr(row, geometry_column)
            },
            TYPE: getattr(row, type_column),
            LABELS: getattr(row, labels_column, []),
            ID: getattr(row, id_column, None)
        }

        return Annotation.from_dict(data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._build_annotation(self._obj.iloc[key])

        return [
            self._build_annotation(row)
            for row in self._obj[key].itertuples()]

    def get(
            self,
            row=None,
            id=None,
            type_column=None,
            geometry_column=None,
            labels_column=None,
            id_column=None):
        if id_column is None:
            id_column = self.id_column

        if row is not None:
            row = self._obj.iloc[row]
        elif id is not None:
            row = self._obj[self._obj[id_column] == id].iloc[0]
        else:
            row = self._obj.iloc[0]

        return self._build_annotation(
            row,
            type_column=type_column,
            geometry_column=geometry_column,
            labels_column=labels_column,
            id_column=id_column)

    def get_activity(self, time_unit=60, time_module=None, target_labels=None, min_t=None, max_t=None, exclude=[]):
        if "abs_start_time" not in self._obj.columns:
            raise ValueError("Annotations should have an absolute time reference in order to compute activity.")

        if min_t is None:
            min_t = self._obj.abs_start_time.min()
        if max_t is None:
            max_t = self._obj.abs_start_time.max()

        if min_t >= max_t:
            raise ValueError("Wrong time range. Try a more accurate specification.")

        dann = self._obj[(pd.to_datetime(self._obj.abs_start_time, utc=True) >= min_t) & (pd.to_datetime(self._obj.abs_start_time, utc=True) <= max_t)]

        total_time = datetime.timedelta.total_seconds(max_t - min_t)
        if time_module is not None:
            module = datetime.timedelta(seconds=time_unit*time_module)
            nframes = time_module
        else:
            module = None
            nframes = int(np.round(total_time/time_unit))

        activities = {}
        if target_labels is None:
            activity = np.zeros([nframes])
            for start_time, end_time, abs_start_time, labels in dann[["start_time", "end_time", "abs_start_time", "labels"]].values:
                toss = False
                for l in labels:
                    if (l["key"], l["value"]) in exclude:
                        toss = True
                if not toss:
                    if time_module is None:
                        start = int(np.round(float(datetime.timedelta.total_seconds(pd.to_datetime(abs_start_time, utc=True) - min_t))/time_unit))
                        stop = max(int(np.round(float(end_time - start_time)/time_unit)), start)
                        activity[start:stop+1] += 1
                    else:
                        remainder = (pd.to_datetime(abs_start_time, utc=True) - min_t.astimezone("utc")) % module
                        index = np.int64(int(round((remainder/time_unit).total_seconds())) % time_module)
                        activity[index] += 1
            activities["Any"] = activity
        else:
            nlabels = len(target_labels)
            for n in range(nlabels):
                label_activity = np.zeros([nframes])
                for start_time, end_time, abs_start_time, labels in dann[["start_time", "end_time", "abs_start_time", "labels"]].values:
                    for l in labels:
                        if l["key"] == target_labels[n]["key"] and l["value"] == target_labels[n]["value"] and (l["key"], l["value"]) not in exclude:
                            if time_module is None:
                                start = int(np.round(float(datetime.timedelta.total_seconds(pd.to_datetime(abs_start_time, utc=True) - min_t))/time_unit))
                                stop = max(int(np.round((end_time - start_time)/time_unit)), start)
                                label_activity[start:stop+1] += 1
                            else:
                                remainder = (pd.to_datetime(abs_start_time, utc=True) - min_t.astimezone("utc")) % module
                                index = np.int64(int(round((remainder/time_unit).total_seconds())) % time_module)
                                label_activity[index] += 1
                activities[target_labels[n]["value"]] = label_activity

        labels = list(activities.keys())
        labels.sort()
        activities["abs_start_time"] = [min_t.astimezone("utc") + datetime.timedelta(seconds=i*time_unit) for i in range(nframes)]
        activities["abs_end_time"] = [min_t.astimezone("utc") + datetime.timedelta(seconds=(i+1)*time_unit) for i in range(nframes)]

        return pd.DataFrame(activities)[["abs_start_time", "abs_end_time"]+labels]

    def change_type_column(self, new_column):
        self.type_column = new_column

    def change_geometry_column(self, new_column):
        self.geometry_column = new_column

    def change_labels_column(self, new_column):
        self.labels_column = new_column

    def change_id_column(self, new_column):
        self.id_column = new_column
