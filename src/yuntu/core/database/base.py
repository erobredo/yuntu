"""Base definition for yuntu databases."""
from collections import namedtuple
from pony.orm import Database
from pony.orm import db_session

from yuntu.core.database.annotations import build_base_annotation_model
from yuntu.core.database.annotations import build_timed_annotation_model
from yuntu.core.database.recordings import build_base_recording_model
from yuntu.core.database.recordings import build_timed_recording_model
from yuntu.core.database.datastores import build_base_datastore_model
from yuntu.core.database.datastores import build_foreign_db_datastore_model
from yuntu.core.database.datastores import build_storage_model
from yuntu.core.database.datastores import build_remote_storage_model

MODELS = [
    'recording',
    'annotation',
    'datastore',
    'foreign_db_store',
    'storage',
    'remote_storage'
]
Models = namedtuple('Models', MODELS)


class DatabaseManager:
    """Base manager for databases.

    Handles database creation, initialization and communication with
    collections.
    """

    def __init__(self, provider, config=None):
        if provider == "irekua":
             raise ValueError("'irekua' provider only works with irekuaREST manager")
        self.provider = provider
        self.config = config
        self.db = Database()

        self.models = self.build_models()
        self.init_db()

    def init_db(self):
        """Initialize database.

        Will bind with database and generate all tables.
        """
        self.db.bind(self.provider, **self.config)
        self.db.generate_mapping(create_tables=True)

    def build_models(self):
        """Construct all database entities."""
        recording = self.build_recording_model()
        datastore = self.build_datastore_model()
        annotation = self.build_annotation_model()

        foreign, storage, remote = self.build_extra_datastores(datastore)
        models = {
            'recording': recording,
            'datastore': datastore,
            'annotation': annotation,
            'foreign_db_store': foreign,
            'storage': storage,
            'remote_storage': remote
        }
        return Models(**models)

    def build_datastore_model(self):
        """Construct the datastore entity."""
        return build_base_datastore_model(self.db)

    def build_recording_model(self):
        """Construct the recording entity."""
        return build_base_recording_model(self.db)

    def build_annotation_model(self):
        """Construct the annotation entity."""
        return build_base_annotation_model(self.db)

    def build_extra_datastores(self, datastore):
        """Build supplemental datastores for specific behaviours."""
        foreign = build_foreign_db_datastore_model(datastore)
        storage = build_storage_model(datastore)
        remote = build_remote_storage_model(storage)
        return foreign, storage, remote

    def get_model_class(self, model):
        """Return model class if exists or raise error."""
        model_dict = self.models._asdict()

        if model not in model_dict:
            options = model_dict.keys()
            options_str = ', '.join(options)
            message = (
                f'The model {model} is not installed in the database. '
                f'Admisible options: {options_str}')
            raise NotImplementedError(message)

        return model_dict[model]

    @db_session
    def select(self, query, model="recording"):
        """Query entries from database."""
        model_class = self.get_model_class(model)

        if query is None:
            return model_class.select()

        return model_class.select(query)

    @db_session
    def delete(self, query, model="recording"):
        """Delete entries using filter."""
        model_class = self.get_model_class(model)
        return [obj.delete() for obj in model_class.select(query)]

    @db_session
    def update(self, query, set_obj, model="recording"):
        """Update matches."""
        model_class = self.get_model_class(model)
        return [obj.set(**set_obj) for obj in model_class.select(query)]

    @db_session
    def insert(self, meta_arr, model="recording"):
        """Directly insert new media entries without a datastore."""
        model_class = self.get_model_class(model)
        return [model_class(**meta) for meta in meta_arr]


class TimedDatabaseManager(DatabaseManager):

    def build_recording_model(self):
        recording = super().build_recording_model()
        return build_timed_recording_model(recording)

    def build_annotation_model(self):
        annotation = super().build_annotation_model()
        return build_timed_annotation_model(annotation)
