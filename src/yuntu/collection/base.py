"""Base classes for collection."""
import os
import json
import pandas as pd

from yuntu.core.database.base import DatabaseManager
from yuntu.core.database.base import TimedDatabaseManager
from yuntu.core.audio.audio import Audio
from yuntu.core.annotation.annotation import Annotation
from yuntu.datastore.copy import CopyDatastore

def _parse_annotation(annotation):
    return {
        'type': annotation.type,
        'id': annotation.id,
        'labels': annotation.labels,
        'metadata': annotation.metadata,
        'geometry': {
            'wkt': annotation.geometry
        }
    }


class Collection:
    """Base class for all collections."""

    db_config = {
        'provider': 'sqlite',
        'config': {
            'filename': ':memory:',
            'create_db': True
        }
    }
    audio_class = Audio
    annotation_class = Annotation
    db_manager_class = DatabaseManager

    def __init__(self, db_config=None, base_path=""):
        """Initialize collection."""
        self.base_path = base_path
        if self.base_path != "":
            if self.base_path[:5] != "s3://":
                if not os.path.isabs(self.base_path):
                    self.base_path = os.path.abspath(self.base_path)

        if db_config is not None:
            self.db_config = db_config

        if self.db_config["provider"] == "sqlite":
            if self.base_path != "":
                filename = self.db_config["config"]["filename"]
                if not os.path.isabs(filename):
                    filename = os.path.join(self.base_path, filename)
                self.db_config["config"]["filename"] = filename

        self.db_manager = self.get_db_manager()

    def __getitem__(self, key):
        queryset = self.recordings()
        if isinstance(key, int):
            return self.build_audio(queryset[key:key + 1][0])

        return [self.build_audio(recording) for recording in queryset[key]]

    def __iter__(self):
        for recording in self.recordings():
            yield self.build_audio(recording)

    def __len__(self):
        return len(self.recordings())

    def get(self, key, with_metadata=True):
        record = self.recordings(lambda rec: rec.id == key).get()
        return self.build_audio(record, with_metadata=with_metadata)

    def get_recording_dataframe(
            self,
            query=None,
            limit=None,
            offset=0,
            with_metadata=False,
            with_annotations=False):
        if limit is None:
            query_slice = slice(offset, None)
        else:
            query_slice = slice(offset, offset + limit)
        recordings = self.recordings(query=query)[query_slice]

        records = []
        for recording in recordings:
            data = recording.to_dict()
            media_info = data.pop('media_info')
            data.update(media_info)
            data["path"] = self.get_abspath(data["path"])

            if not with_metadata:
                data.pop('metadata')

            if with_annotations:
                data['annotations'] = [
                    _parse_annotation(annotation)
                    for annotation in recording.annotations]

            records.append(data)

        return pd.DataFrame(records)

    def get_annotation_dataframe(
            self,
            query=None,
            limit=None,
            offset=0,
            with_metadata=None):
        if limit is None:
            query_slice = slice(offset, None)
        else:
            query_slice = slice(offset, offset + limit)
        annotations = self.annotations(query=query)[query_slice]

        records = []
        for annotation in annotations:
            data = annotation.to_dict()
            labels = data.pop('labels')

            if not with_metadata:
                data.pop('metadata')

            data['labels'] = labels

            for label in labels:
                data[label['key']] = label['value']

            records.append(data)

        return pd.DataFrame(records)

    def get_db_manager(self):
        return self.db_manager_class(**self.db_config)

    def get_abspath(self, path):
        if path[:5] != "s3://":
            if not os.path.isabs(path):
                return os.path.join(self.base_path, path)
        return path

    def insert(self, meta_arr):
        """Directly insert new media entries without a datastore."""
        if not isinstance(meta_arr, (list, tuple)):
            meta_arr = [meta_arr]
        return self.db_manager.insert(meta_arr)

    def annotate(self, meta_arr):
        """Insert annotations to database."""
        return self.db_manager.insert(meta_arr, model="annotation")

    def update_recordings(self, query, set_obj):
        """Update matches."""
        return self.db_manager.update(query, set_obj, model="recordings")

    def update_annotations(self, query, set_obj):
        """Update matches."""
        return self.db_manager.update(query, set_obj, model="annotations")

    def delete_recordings(self, query):
        """Delete matches."""
        return self.db_manager.delete(query, model='recording')

    def delete_annotations(self, query):
        """Delete matches."""
        return self.db_manager.delete(query, model='annotation')

    @property
    def recordings_model(self):
        return self.db_manager.models.recording

    @property
    def annotations_model(self):
        return self.db_manager.models.annotation

    def annotations(self, query=None, iterate=True):
        """Retrieve annotations from database."""
        matches = self.db_manager.select(query, model="annotation")
        if iterate:
            return matches
        return list(matches)

    def recordings(self, query=None, iterate=True):
        """Retrieve audio objects."""
        matches = self.db_manager.select(query, model="recording")
        if iterate:
            return matches
        return list(matches)

    def build_audio(self, recording, with_metadata=True):
        annotations = []
        for annotation in recording.annotations:
            data = annotation.to_dict()
            annotation = self.annotation_class.from_record(data)
            annotations.append(annotation)

        metadata = recording.metadata if with_metadata else None

        path = self.get_abspath(recording.path)

        return self.audio_class(
            path=path,
            id=recording.id,
            media_info=recording.media_info,
            timeexp=recording.timeexp,
            metadata=metadata,
            annotations=annotations,
            lazy=True)

    def pull(self, datastore):
        """Pull data from datastore and insert into collection."""
        datastore.insert_into(self)

    def materialize(self, out_name, query=None, out_dir="", tqdm=None):
        """Create materialized collection."""
        target_path = os.path.join(out_dir, out_name)
        copystore = CopyDatastore(collection=self,
                                  query=query,
                                  target_path=target_path,
                                  tqdm=tqdm)
        col_type = "simple"
        if isinstance(self.db_manager, TimedDatabaseManager):
            col_type = "timed"

        target_config = {
            "col_type": col_type,
            "db_config": {
                "provider": "sqlite",
                "config": {
                    "filename": "db.sqlite",
                    "create_db": True
                }
            }
        }

        col_config_path = os.path.join(target_path, f"col_config.json")

        with open(col_config_path, "w") as f:
            json.dump(target_config, f)

        target_col = self.__class__(db_config=target_config["db_config"],
                                    base_path=target_path)

        _, _, _, = copystore.insert_into(target_col)

        return target_path

class TimedCollection(Collection):
    """Time aware collection."""
    db_manager_class = TimedDatabaseManager
