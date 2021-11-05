"""Base classes for probes that use tflite models as criteria."""
import os
import numpy as np
from abc import ABC
from yuntu.soundscape.processors.probes.base import ModelProbe

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

class TFLiteModelProbe(ModelProbe, ABC):
    """A model probe that uses keras."""
    def __init__(self, model_path):
        self._model = None
        self.model_path = model_path
        self._input_indices = None
        self._output_indices = None

    def get_output(self, inputs, output_indices=[0]):
        for i in range(len(inputs)):
            self.model.set_tensor(self._input_indices[i], np.array(inputs[i]["value"], dtype=inputs[i]["dtype"]))
        self.model.invoke()
        predictions = []

        for i in range(len(output_indices)):
            pred = self.model.get_tensor(self._output_indices[output_indices[i]])[0]
            predictions.append(pred)

        return predictions

    def load_model(self):
        """Load model from model path."""
        self._model = tflite.Interpreter(model_path=self.model_path)
        self._model.allocate_tensors()
        input_details = self._model.get_input_details()
        output_details = self._model.get_output_details()
        self._input_indices = [input_details[i]["index"]
                               for i in range(len(input_details))]
        self._output_indices = [output_details[i]["index"]
                               for i in range(len(output_details))]

    def clean(self):
        """Remove memory footprint."""
        del self._model
