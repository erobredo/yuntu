from yuntu.core.pipeline.base import Pipeline
from yuntu.core.pipeline.places.extended import place

from yuntu.soundscape.transitions.basic import get_partitions
from yuntu.soundscape.transitions.probe import probe_annotate


class ProbeAnnotate(Pipeline):
    """Pipeline that applies probes and directly annotate collections."""

    def __init__(self,
                 name,
                 probe_config,
                 collection_config,
                 query=None,
                 **kwargs):

        if not isinstance(collection_config, dict):
            raise ValueError("Argument 'collection_config' must be a dictionary.")
        if not isinstance(probe_config, dict):
            raise ValueError("Argument 'probe_config' must be a dictionary.")

        super().__init__(name, **kwargs)

        self.query = query
        self.collection_config = collection_config
        self.probe_config = probe_config
        self.build()

    def build(self):
        self["col_config"] = place(self.collection_config, 'dict', 'col_config')
        self["query"] = place(self.query, 'dynamic', 'query')
        self["npartitions"] = place(1, 'scalar', 'npartitions')
        self["probe_config"] = place(self.probe_config, 'dict', 'probe_config')
        self["partitions"] = get_partitions(self["col_config"],
                                            self["query"],
                                            self["npartitions"])
        self["annotation_result"] = probe_annotate(self["partitions"],
                                                   self["probe_config"],
                                                   self["col_config"])
