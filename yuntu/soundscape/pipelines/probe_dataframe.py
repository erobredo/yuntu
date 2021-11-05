from yuntu.core.pipeline.base import Pipeline
from yuntu.core.pipeline.places.extended import place
from yuntu.soundscape.transitions.basic import bag_dataframe
from yuntu.soundscape.transitions.probe import probe_recordings

class ProbeDataframe(Pipeline):
    """Pipeline to apply probe using dask."""

    def __init__(self,
                 name,
                 recordings,
                 probe_config,
                 **kwargs):

        if not isinstance(probe_config, dict):
            raise ValueError("Argument 'probe_config' must be a dictionary.")

        super().__init__(name, **kwargs)

        self.recordings = recordings
        self.probe_config = probe_config
        self.build()

    def build(self):
        self['recordings'] = place(data=self.recordings,
                                   name='recordings',
                                   ptype='pandas_dataframe')
        self['npartitions'] = place(data=10,
                                    name='npartitions',
                                    ptype='scalar')
        self["probe_config"] = place(self.probe_config, 'dict', 'probe_config')
        self['recordings_bag'] = bag_dataframe(self['recordings'],
                                               self['npartitions'])
        self["matches"] = probe_recordings(self["recordings_bag"],
                                           self["probe_config"])
