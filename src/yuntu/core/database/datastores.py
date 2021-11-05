"""Distinct types of datastores."""
from datetime import datetime
from pony.orm import Required
from pony.orm import Optional
from pony.orm import PrimaryKey
from pony.orm import Set
from pony.orm import Json


def build_base_datastore_model(db):
    """Create base datastore model."""
    class Datastore(db.Entity):
        """Basic datastore entity for yuntu."""
        id = PrimaryKey(int, auto=True)
        file = Optional(str)
        recordings = Set('Recording')
        metadata = Required(Json)

    return Datastore


def build_foreign_db_datastore_model(Datastore):
    class ForeignDb(Datastore):
        """Datastore that builds data from a foreign database."""
        host = Required(str)
        database = Required(str)
        query = Required(str)
        last_update = Required(datetime)
    return ForeignDb


def build_storage_model(Datastore):
    class Storage(Datastore):
        """Datastore builds from directory structure."""
        dir_path = Required(str)
    return Storage


def build_remote_storage_model(Storage):
    class RemoteStorage(Storage):
        """Datastore that builds metadata from a remote storage."""
        metadata_url = Optional(str)
    return RemoteStorage
