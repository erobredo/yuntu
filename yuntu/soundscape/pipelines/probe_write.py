from yuntu.core.pipeline.base import Pipeline
from yuntu.core.pipeline.places.extended import place

from yuntu.soundscape.transitions.basic import get_partitions
from yuntu.soundscape.transitions.probe import probe_write


class ProbeWrite(Pipeline):
    """Pipeline to build tfrecords from bat call annotations."""

    def __init__(self,
                 name,
                 probe_config,
                 collection_config,
                 write_config,
                 query=None,
                 **kwargs):
        if not isinstance(collection_config, dict):
            raise ValueError("Argument 'collection_config' must be a dictionary.")
        if not isinstance(write_config, dict):
            raise ValueError("Argument 'write_config' must be a dictionary.")
        if not isinstance(probe_config, dict):
            raise ValueError("Argument 'probe_config' must be a dictionary.")

        super().__init__(name, **kwargs)

        self.query = query
        self.collection_config = collection_config
        self.write_config = write_config
        self.probe_config = probe_config
        self.build()

    def build(self):
        self["col_config"] = place(self.collection_config, 'dict', 'col_config')
        self["query"] = place(self.query, 'dynamic', 'query')
        self["npartitions"] = place(1, 'scalar', 'npartitions')
        self["write_config"] = place(self.write_config, "dict", "write_config")
        self["overwrite"] = place(False, "dynamic", "overwrite")
        self["batch_size"] = place(200, 'scalar', 'batch_size')
        self["probe_config"] = place(self.probe_config, 'dict', 'probe_config')

        self["partitions"] = get_partitions(self["col_config"],
                                            self["query"],
                                            self["npartitions"])
        self["write_result"] = probe_write(self["partitions"],
                                           self["probe_config"],
                                           self["col_config"],
                                           self["write_config"],
                                           self["batch_size"],
                                           self["overwrite"])
