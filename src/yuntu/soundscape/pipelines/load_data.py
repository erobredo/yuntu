from yuntu.core.pipeline.base import Pipeline
from yuntu.core.pipeline.places.extended import place

from yuntu.soundscape.transitions.basic import load_datastores
from yuntu.soundscape.transitions.basic import pg_init_database
from yuntu.soundscape.transitions.basic import source_partition

class DatastoreLoad(Pipeline):
    """Pipeline to initialize a collection and insert data from datastores."""
    _partitions = None

    def __init__(self,
                 name,
                 datastore_configs,
                 collection_config,
                 **kwargs):
        if not isinstance(collection_config, dict):
            raise ValueError(
                "Argument 'collection_config' must be a dictionary.")
        if not isinstance(datastore_configs, list):
            raise ValueError(
                "Argument 'datastore_configs' must be a list.")

        super().__init__(name, **kwargs)

        self.collection_config = collection_config
        self.datastore_configs = datastore_configs
        self.build()

    def build(self):
        self["col_config"] = place(self.collection_config, 'dict', 'col_config')
        self["datastore_configs"] = place(self.datastore_configs, 'dynamic', 'datastore_configs')
        self["insert_results"] = load_datastores(self["col_config"],
                                                 self["datastore_configs"])


class DatastoreLoadPartitioned(Pipeline):
    """Pipeline to initialize a collection and insert data from datastores."""
    _partitions = None

    def __init__(self,
                 name,
                 datastore_config,
                 collection_config,
                 admin_config,
                 rest_auth,
                 **kwargs):
        if not isinstance(collection_config, dict):
            raise ValueError(
                "Argument 'collection_config' must be a dictionary.")
        if not isinstance(admin_config, dict):
            raise ValueError(
                "Argument 'admin_config' must be a dictionary.")
        if not isinstance(datastore_config, dict):
            raise ValueError(
                "Argument 'datastore_configs' must be a dict.")

        super().__init__(name, **kwargs)

        self.admin_config = admin_config
        self.collection_config = collection_config
        self.datastore_config = datastore_config
        self.rest_auth = rest_auth
        self.build()

    def build(self):
        self["init_config"] = place(self.collection_config, 'dict', 'init_config')
        self["admin_config"] = place(self.admin_config, 'dynamic', 'admin_config')
        self["datastore_config"] = place(self.datastore_config, 'dynamic', 'datastore_config')
        self['npartitions'] = place(1, 'scalar', 'npartitions')
        self["rest_auth"] = place(self.rest_auth, 'dynamic', 'rest_auth')
        self["datastore_configs"] = source_partition(self["datastore_config"],
                                                     self["rest_auth"],
                                                     self['npartitions'])
        self["col_config"] = pg_init_database(self["init_config"],
                                              self["admin_config"])
        self["insert_results"] = load_datastores(self["col_config"],
                                                 self["datastore_configs"])
