import pandas as pd
from yuntu.core.audio.audio import Audio
from yuntu.collection.base import Collection
from yuntu.core.database.REST.irekua import IrekuaREST

class IrekuaRESTCollection(Collection):
    """Base class for all collections."""

    db_config = {
        'provider': 'irekua',
        'config': {
            'api_url': 'http://localhost:3000/api/',
            'version': 'v1',
            'page_size': 1000,
            'target_attr': 'results',
            'auth': 'abc:xyz',
            'bucket': None,
            'base_filter': None
        }
    }

    db_manager_class = IrekuaREST

    def __init__(self, db_config=None):
        """Initialize collection."""
        if db_config is not None:
            self.db_config = db_config

        self.db_manager = self.get_db_manager()

    def __getitem__(self, key):

        if isinstance(key, int):
            matches = self.recordings(limit=1, offset=key, iterate=False)
            if len(matches) == 0:
                raise ValueError(f"Recording {key} not found.")
            return self.build_audio(matches[0])

        limit = None
        offset = None
        if key.start is not None:
            offset = key.start
            if key.stop is not None:
                limit = key.stop - key.start
        elif key.stop is not None:
            limit = key.stop

        if limit is not None:
            if limit < 0:
                raise ValueError("Wrong keys.")
            elif limit == 0:
                return []

        return ([self.build_audio(recording)
                for recording in self.recordings(limit=limit,
                                                 offset=offset)])
    def __iter__(self):
        for recording in self.recordings():
            yield self.build_audio(recording)

    def __len__(self):
        return self.recordings_model.count()

    def get(self, key, with_metadata=True):
        matches = self.recordings(limit=1, offset=key, iterate=False)
        if len(matches) == 0:
            raise ValueError(f"Recording {key} not found.")
        return self.build_audio(matches[0], with_metadata=with_metadata)

    def get_recording_dataframe(
            self,
            query=None,
            limit=None,
            offset=None,
            with_metadata=False,
            fetch_meta=[]):
        
        if not with_metadata:
            fetch_meta = []

        recordings = self.recordings(query=query,
                                     limit=limit,
                                     offset=offset,
                                     fetch_meta=fetch_meta)

        records = []
        for recording in recordings:
            data = recording.to_dict()
            media_info = data.pop('media_info')
            data.update(media_info)

            if not with_metadata:
                data.pop('metadata')

            records.append(data)

        return pd.DataFrame(records)

    def get_annotation_dataframe(self,
                                 query=None,
                                 limit=None,
                                 offset=0,
                                 with_metadata=None):
        pass

    def get_db_manager(self):
        return self.db_manager_class(**self.db_config)

    def insert(self, meta_arr):
        """Directly insert new media entries without a datastore."""
        pass

    def annotate(self, meta_arr):
        """Insert annotations to database."""
        pass

    def update_recordings(self, query, set_obj):
        """Update matches."""
        pass

    def update_annotations(self, query, set_obj):
        """Update matches."""
        pass

    def delete_recordings(self, query):
        """Delete matches."""
        pass

    def delete_annotations(self, query):
        """Delete matches."""
        pass

    @property
    def annotations_model(self):
        return None

    def annotations(self, query=None, iterate=True):
        """Retrieve annotations from database."""
        pass

    def recordings(self, query=None, limit=None, offset=None, iterate=True, fetch_meta=[]):
        """Retrieve audio objects."""
        matches = self.db_manager.select(query, limit=limit, offset=offset, model="recording", fetch_meta=fetch_meta)
        if iterate:
            return matches
        return list(matches)

    def build_audio(self, recording, with_metadata=True):
        annotations = []

        metadata = recording.metadata if with_metadata else None

        return self.audio_class(
            path=recording.path,
            id=recording.id,
            media_info=recording.media_info,
            timeexp=recording.timeexp,
            metadata=metadata,
            annotations=annotations,
            lazy=True)

    def pull(self, datastore):
        """Pull data from datastore and insert into collection."""
        pass
