"""Base classes for probes that use keras models as criteria."""
from abc import ABC
import tensorflow.keras as keras

from yuntu.soundscape.processors.probes.base import ModelProbe

class KerasModelProbe(ModelProbe, ABC):
    """A model probe that uses keras."""

    def load_model(self):
        """Load model from model path."""
        self._model = keras.models.load_model(self.model_path, compile=False)

    def clean(self):
        """Remove memory footprint."""
        del self._model
        keras.backend.clear_session()
