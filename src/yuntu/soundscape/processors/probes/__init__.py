"""Soundscape probes modules."""
from yuntu.soundscape.processors.probes.base import Probe, TemplateProbe, ModelProbe
from yuntu.soundscape.processors.probes.crosscorr import CrossCorrelationProbe
# Adding the following line imposes tensorflow as a dependency
# from yuntu.soundscape.processors.probes.keras import KerasModelProbe
# from yuntu.soundscape.processors.probes.tflite import TFLiteModelProbe
from yuntu.soundscape.processors.probes.methods import probe

__all__ = [
    'Probe',
    'TemplateProbe',
    'ModelProbe',
    'CrossCorrelationProbe',
    'probe'
]
