"""Audio dataframe base classes.

An audio dataframe is a
"""
import numpy as np
import pandas as pd
import datetime
import shapely.wkt
from shapely.ops import unary_union
from shapely.geometry import Polygon, box

from yuntu.core.utils.atlas import buffer_geometry
from yuntu.core.annotation.labels import Labels
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

SINGLE_COUNTER = lambda x: x+1
BOOLEAN_COUNTER = lambda x: np.maximum(x, np.ones_like(x))
DEFAULT_COUNTER = SINGLE_COUNTER

def buffer_geometry_clip(geom, radius):
    '''Buffers geometry clipping to safe ranges'''
    if not isinstance(geom, Polygon):
        raise NotImplementedError(f"Geometry type not supported. Only polygons can be disolved for now.")

    time_radius, freq_radius = radius
    radius = [time_radius, max(1e-10, freq_radius)]
    bgeom = buffer_geometry(geom, radius)
    x, y = bgeom.exterior.coords.xy
    y = np.clip(y, 0, 1e16)
    x = np.clip(x, 0, None)
    return Polygon(zip(x,y)).simplify(0)

def disolve_annotations(group, key, radius, join_meta_func=None, keep_radius=True):
    '''Return disolved annotations within group'''

    recording, label_str, dtype, classtype = group.name
    ori_geoms = group[["id", "labels", "metadata"]]
    ori_geoms.loc[:, "geometry"] = group.geometry.apply(lambda x: shapely.wkt.loads(x))

    if radius is not None:
        ref_geometries = [buffer_geometry_clip(x, radius) for x in ori_geoms.geometry.values]
    else:
        ref_geometries = [x for x in ori_geoms.geometry.values]

    ref_geometries = unary_union(ref_geometries)

    if classtype == "TimedAnnotation":
        min_abs_start_time = group.abs_start_time.min()
        min_start_time = group.start_time.min()

    rows = []
    geoms = ref_geometries
    if isinstance(geoms, Polygon):
        geoms = [geoms]

    for geom in geoms:
        row = {}
        bgeom = geom.buffer(1e-10)
        members = ori_geoms[ori_geoms.geometry.apply(lambda x: x.within(bgeom))]
        member_ids = list(members.id.values.astype(int))
        labels = None
        for n, lab in members.labels.items():
            llab = Labels.from_dict(lab)
            if labels is None:
                labels = llab
                continue
            for l in llab:
                if l.key not in labels:
                    labels.add(l)

        if join_meta_func is not None:
            metadata = join_meta_func(members.metadata.values)
        else:
            metadata = {}

        metadata["disolve"] = {
            "members": member_ids,
            "group": {"key": key,
                      "value": label_str,
                      "radius": radius,
                      "keep_radius": keep_radius}
        }

        if not keep_radius and radius is not None:
            if len(member_ids) == 1:
                geometry = members.iloc[0].geometry
            else:
                geometry = geom.intersection(box(*unary_union(members.geometry.values).bounds, ccw=True))
        else:
            geometry = geom

        start_time, min_freq, end_time, max_freq = geometry.bounds

        labels = labels.to_dict()
        row["geometry"] = geometry.wkt
        row["start_time"] = start_time
        row["end_time"] = end_time
        row["max_freq"] = max_freq
        row["min_freq"] = min_freq

        if classtype == "TimedAnnotation":
            row["abs_start_time"] = min_abs_start_time + datetime.timedelta(seconds=start_time-min_start_time)
            row["abs_end_time"] = min_abs_start_time + datetime.timedelta(seconds=end_time-min_start_time)

        row["labels"] = labels
        row["metadata"] = metadata
        row["labels"] = labels
        rows.append(row)

    return pd.DataFrame(rows)

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

    def disolve(self, key, radius=(1.5, 0), join_meta_func=None, keep_radius=True):
        '''Merge annotations by geometry and key'''

        if len(self._obj.classtype.unique()) > 1:
            raise ValueError("Can only disolve dataframes with homogeneous classtypes for now. Filter by classtype first.")

        return (self._obj
                .groupby(by=["recording", key, "type", "classtype"], as_index=True)
                .apply(disolve_annotations, radius=radius, key=key, join_meta_func=join_meta_func, keep_radius=keep_radius)
                .reset_index(level=-1, drop=True)
                .reset_index())

    def get_activity(self, count_func=DEFAULT_COUNTER, time_unit=60, time_module=None, target_labels=None, min_t=None, max_t=None, exclude=[]):
        if "abs_start_time" not in self._obj.columns:
            raise ValueError("Annotations should have an absolute time reference in order to compute activity.")

        if min_t is None:
            min_t = pd.to_datetime(self._obj.abs_start_time.min(), utc=True)
        if max_t is None:
            max_t = pd.to_datetime(self._obj.abs_end_time.max(), utc=True)

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
            if nframes == 0 and not dann.empty:
                nframes = 1

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
                        activity[start:stop+1] = count_func(activity[start:stop+1])
                    else:
                        remainder = (pd.to_datetime(abs_start_time, utc=True) - min_t.astimezone("utc")) % module
                        index = np.int64(int(round((remainder/time_unit).total_seconds())) % time_module)
                        activity[index] = count_func(activity[index])
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
