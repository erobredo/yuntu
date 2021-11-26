import os
from pony.orm import db_session
from yuntu.datastore.base import Datastore
from yuntu.core.audio.utils import media_copy
from yuntu.core.database.timed import TimedDatabaseManager
from yuntu.core.database.spatial import SpatialDatabaseManager
from yuntu.core.database.spatiotemporal import SpatioTemporalDatabaseManager

class CopyDatastore(Datastore):
    """A datastore that copies all files to a target_directory and inserts
    metadata into a new collection"""

    def __init__(self, *args, collection, query=None, limit=None, offset=0,
                 target_path=None, keep_metadata=True, absolute_path=False,
                 keep_id=True, tqdm=None, **kwargs):
        self.limit = limit
        self.offset = offset
        self.query = query
        self.absolute_path = absolute_path
        self.target_path = target_path
        self.collection = collection
        self.media_path = None
        self.keep_metadata = keep_metadata
        self.keep_id = keep_id
        self.tqdm = tqdm

        if self.offset is not None:
            if self.limit is None:
                self.query_slice = slice(self.offset, None)
            else:
                self.query_slice = slice(self.offset, self.offset + self.limit)

        if self.target_path is not None:
            self.init_target()

        super().__init__(*args, **kwargs)

    def init_target(self):
        media_path = os.path.join(self.target_path, "media")
        if not os.path.exists(media_path):
            os.makedirs(media_path)
        self.media_path = media_path

    @property
    def size(self):
        if self._size is None:
            self._size = len(self.collection.recordings(query=self.query)[self.query_slice])
        return self._size

    def get_metadata(self):
        meta = {"type": "CopyDatastore"}

        col_type = "simple"
        if isinstance(self.collection.db_manager, TimedDatabaseManager):
            col_type = "timed"
        elif isinstance(self.collection.db_manager, SpatialDatabaseManager):
            col_type = "spatial"
        elif isinstance(self.collection.db_manager, SpatioTemporalDatabaseManager):
            col_type = "spatiotemporal"

        col_config = {
            "col_type": col_type,
            "db_config": self.collection.db_config
        }

        meta["col_config"] = col_config
        meta["query"] = str(self.query)
        return meta

    def iter(self):
        if self.tqdm is not None:
            with self.tqdm(total=self.size) as pbar:
                for recording in self.collection.recordings(query=self.query)[self.query_slice]:
                    pbar.update(1)
                    yield recording
        else:
            for recording in self.collection.recordings(query=self.query)[self.query_slice]:
                yield recording

    def iter_annotations(self, datum):
        recid = datum.id
        query = eval(f'lambda annotation: annotation.recording.id == {recid}')
        for annotation in self.collection.annotations(query=query):
            yield annotation

    def copy_data(self, datum):
        source_path = self.collection.get_abspath(datum.path)
        recid = datum.id
        target_path = os.path.join(self.media_path, f"{recid}_copy.wav")
        media_copy(source_path, target_path)

        if not self.absolute_path:
            return os.path.join("media", f"{recid}_copy.wav")

        return os.path.abspath(target_path)

    def prepare_datum(self, datum):
        meta = datum.to_dict()
        if self.target_path is not None:
            path = self.copy_data(datum)
        else:
            path = meta["path"]

        meta["path"] = path

        if not self.keep_metadata:
            meta["metadata"] = {}
        else:
            meta["metadata"] = dict(meta["metadata"])

        meta["metadata"]["yuntu_foreign_id"] = {"foreign_id": meta["id"],
                                                "foreign_datastore_id": meta["datastore"]}
        del meta["datastore"]

        if not self.keep_id:
            del meta["id"]

        return meta

    def prepare_annotation(self, datum, annotation):
        meta = dict(annotation.to_dict())
        del meta["recording"]
        return meta

    def insert_into(self, collection):
        datastore_record = self.create_datastore_record(collection)
        datastore_record.flush()
        datastore_id = datastore_record.id

        recording_inserts = 0
        annotation_inserts = 0
        for datum in self.iter():
            meta = self.prepare_datum(datum)
            meta['datastore'] = datastore_id

            annotations = []
            for annotation in self.iter_annotations(datum):
                annotation_meta = self.prepare_annotation(datum, annotation)
                annotations.append(annotation_meta)

            with db_session:
                recording = collection.insert(meta)[0]
                for ann in annotations:
                    ann["recording"] = recording
                    collection.annotate([ann])
                    annotation_inserts += 1

            recording_inserts += 1

        return datastore_id, recording_inserts, annotation_inserts
