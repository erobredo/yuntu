"""Soundscape base pipeline."""
from yuntu.core.audio.features.spectrogram import N_FFT
from yuntu.core.audio.features.spectrogram import HOP_LENGTH
from yuntu.core.audio.features.spectrogram import WINDOW_FUNCTION
from yuntu.core.pipeline.base import Pipeline
from yuntu.core.pipeline.places.extended import place

from yuntu.soundscape.processors.indices.direct import EXAG
from yuntu.soundscape.processors.indices.direct import INFORMATION
from yuntu.soundscape.processors.indices.direct import CORE
from yuntu.soundscape.processors.indices.direct import TOTAL

from yuntu.soundscape.transitions.basic import as_dd, add_hash, add_absoute_time
from yuntu.soundscape.transitions.index import slice_features

INDICES = [TOTAL(), EXAG(), INFORMATION(), CORE()]
TIME_UNIT = 60
TIME_HOP = 1.0
FREQUENCY_BINS = 100
FREQUENCY_LIMITS = (0, 10000)
FREQUENCY_HOP = 1.0
FEATURE_TYPE = 'spectrogram'
FEATURE_CONFIG = {"n_fft": N_FFT,
                  "hop_length": HOP_LENGTH,
                  "window_function": WINDOW_FUNCTION}
TIME_UTC_COLUMN = "time_utc"
HASHER_CONFIG = {
    "module":{
        "object_name": "yuntu.soundscape.hashers.crono.CronoHasher"
    },
    "kwargs": {"time_utc_column": "abs_start_time"}
}
HASH_NAME = 'crono_hash'

class Soundscape(Pipeline):
    """Basic soundscape pipeline"""

    def __init__(self,
                 name="get_soundscape",
                 recordings=None,
                 indices=INDICES,
                 time_unit=TIME_UNIT,
                 frequency_bins=FREQUENCY_BINS,
                 frequency_limits=FREQUENCY_LIMITS,
                 feature_type=FEATURE_TYPE,
                 feature_config=FEATURE_CONFIG,
                 time_hop=TIME_HOP,
                 frequency_hop=FREQUENCY_HOP,
                 **kwargs):
        super().__init__(name, **kwargs)
        if not isinstance(indices, (tuple, list)):
            message = "Argument 'indices' must be a tuple or a list of " + \
                      " acoustic indices."
            raise ValueError(message)
        self.recordings = recordings
        self.indices = indices
        self.time_unit = time_unit
        self.frequency_bins = frequency_bins
        self.frequency_limits = frequency_limits
        self.feature_type = feature_type
        self.feature_config = feature_config
        self.frequency_hop = frequency_hop
        self.time_hop = time_hop
        self.build()

    def build(self):
        """Build soundscape processing pipeline."""
        slice_config_dict = {'time_unit': self.time_unit,
                             'frequency_bins': self.frequency_bins,
                             'frequency_limits': self.frequency_limits,
                             'feature_type': self.feature_type,
                             'feature_config': self.feature_config,
                             'frequency_hop': self.frequency_hop,
                             'time_hop': self.time_hop}
        self['slice_config'] = place(data=slice_config_dict,
                                     name='slice_config',
                                     ptype='dict')
        self['recordings'] = place(data=self.recordings,
                                   name='recordings',
                                   ptype='pandas_dataframe')
        self['indices'] = place(data=self.indices,
                                name='indices',
                                ptype='pickleable')
        self['npartitions'] = place(data=10,
                                    name='npartitions',
                                    ptype='scalar')
        self['recordings_dd'] = as_dd(self['recordings'],
                                      self['npartitions'])
        self['soundscape'] = slice_features(self['recordings_dd'],
                                            self['slice_config'],
                                            self['indices'])


class AbsoluteTimeSoundscape(Pipeline):
    """Timed soundscape pipeline.

    Adds absolute timing from local reference to file.
    """
    def __init__(self,
                 name="abs_time_soundscape",
                 soundscape_pd=None,
                 time_utc_column=TIME_UTC_COLUMN,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.soundscape_pd = soundscape_pd
        self.time_utc_column = time_utc_column
        self.build()

    def build(self):
        """Build soundscape processing pipeline."""
        self['soundscape_pd'] = place(data=self.soundscape_pd,
                                      name='soundscape_pd',
                                      ptype='pandas_dataframe')
        self['npartitions'] = place(data=10,
                                    name='npartitions',
                                    ptype='scalar')
        self['soundscape'] = as_dd(self['soundscape_pd'],
                                   self['npartitions'])
        self['time_utc_column'] = place(data=self.time_utc_column,
                                        name="time_utc_column",
                                        ptype='scalar')
        self['absolute_timed_soundscape'] = add_absoute_time(self['soundscape'],
                                                             self['time_utc_column'])

class HashSoundscape(Pipeline):
    """Hash soundscape pipeline.

    A hashed soundscape is a soundscape that has a special column 'hash'
    indicating some kind of aggregation criteria generated by a Hasher object.
    This pipeline adds a hash using a hasher config.
    """
    def __init__(self,
                 name="hash_soundscape",
                 soundscape_pd=None,
                 hasher_config=HASHER_CONFIG,
                 hash_name=HASH_NAME,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.soundscape_pd = soundscape_pd
        self.hasher_config = hasher_config
        self.hash_name = hash_name
        self.build()

    def build(self):
        """Build soundscape processing pipeline."""
        self['absolute_timed_soundscape_pd'] = place(data=self.soundscape_pd,
                                      name='absolute_timed_soundscape_pd',
                                      ptype='pandas_dataframe')
        self['npartitions'] = place(data=10,
                                    name='npartitions',
                                    ptype='scalar')
        self['absolute_timed_soundscape'] = as_dd(self['absolute_timed_soundscape_pd'],
                                            self['npartitions'])
        self['hasher_config'] = place(data=self.hasher_config,
                                      name='hasher',
                                      ptype='pickleable')
        self['hash_name'] = place(data=self.hash_name,
                                  name="hash_name",
                                  ptype='scalar')
        self['hashed_soundscape'] = add_hash(self['absolute_timed_soundscape'],
                                             self['hasher_config'],
                                             self['hash_name'])

class CronoSoundscape(Soundscape):
    """Full cronological soundscape pipeline.

    Build soundscape, add absolute timing and hash.
    """
    def __init__(self,
                 name="full_soundscape",
                 time_utc_column=TIME_UTC_COLUMN,
                 hasher_config=HASHER_CONFIG,
                 hash_name=HASH_NAME,
                 **kwargs):
        self.time_utc_column = time_utc_column
        self.hasher_config = hasher_config
        self.hash_name = hash_name
        super().__init__(name, **kwargs)

    def build(self):
        """Build soundscape processing pipeline."""
        super().build()
        self['time_utc_column'] = place(data=self.time_utc_column,
                                        name="time_utc_column",
                                        ptype='scalar')
        self['absolute_timed_soundscape'] = add_absoute_time(self['soundscape'],
                                                             self['time_utc_column'])
        self['hasher_config'] = place(data=self.hasher_config,
                                      name='hasher',
                                      ptype='pickleable')
        self['hash_name'] = place(data=self.hash_name,
                                  name="hash_name",
                                  ptype='scalar')
        self['hashed_soundscape'] = add_hash(self['absolute_timed_soundscape'],
                                             self['hasher_config'],
                                             self['hash_name'])
