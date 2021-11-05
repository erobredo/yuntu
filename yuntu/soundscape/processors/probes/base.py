"""Classes for audio probes."""
from abc import ABC
from abc import abstractmethod
import datetime

class Probe(ABC):
    """Base class for all probes.

    Given a signal, a probe is a method that tests matching
    criteria against it and returns a list of media slices that
    satisfy them.
    """

    @abstractmethod
    def apply(self, target, **kwargs):
        """Apply probe and output a list of dicts with a geometry attribute.

        The output should be a list of dictionaries with the following minimal
        structure and information:
        {
            'geometry': <yuntu.core.geometry.Geometry>,
            'labels': [{"key": <str>, "value": <str>, "type": <str>}, ...],
            'score': <dict>
        }

        """

    @abstractmethod
    def clean(self):
        """Remove memory footprint."""

    def prepare_annotation(self, output):
        """Buid annotation from individual output"""

        geom = output["geometry"]
        start_time, min_freq, end_time, max_freq = geom.geometry.bounds
        wkt = geom.geometry.wkt
        meta = {"score": output["score"]}
        geom_type = str(geom.name).replace("Types.", "")

        return {
            "labels": output["labels"],
            "type": f"{geom_type}Annotation",
            "start_time": start_time,
            "end_time": end_time,
            "max_freq": max_freq,
            "min_freq": min_freq,
            "geometry": wkt,
            "metadata": meta
        }

    @property
    def info(self):
        return {
            'probe_class': self.__class__.__name__
        }

    def annotate(self, target, record_time=None, **kwargs):
        """Apply probe and produce annotations"""

        outputs = self.apply(target, **kwargs)
        annotations = []
        for o in outputs:
            ann_dict = self.prepare_annotation(o)

            if "metadata" not in ann_dict:
                ann_dict["metadata"] = {}

            if record_time is not None:
                ann_dict["abs_start_time"] = record_time + datetime.timedelta(seconds=ann_dict["start_time"])
                ann_dict["abs_end_time"] = record_time + datetime.timedelta(seconds=ann_dict["end_time"])

            ann_dict["metadata"]["probe_info"] = self.info
            ann_dict["metadata"]["probe_info"]["kwargs"] = kwargs
            annotations.append(ann_dict)

        return annotations

    def __enter__(self):
        """Behaviour for context manager"""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Behaviour for context manager"""
        self.clean()

    def __call__(self, target, **kwargs):
        """Call apply method."""
        return self.apply(target, **kwargs)

class TemplateProbe(Probe, ABC):
    """A probe that uses a template to find similar matches."""

    @property
    @abstractmethod
    def template(self):
        """Return probe's template."""

    @abstractmethod
    def compare(self, target):
        """Compare target with self's template."""

class ModelProbe(Probe, ABC):
    """A probe that uses any kind of detection or multilabelling model"""

    def __init__(self, model_path):
        self._model = None
        self.model_path = model_path

    @property
    def info(self):
        return {
            'probe_class': self.__class__.__name__,
            'model_path': self.model_path
        }

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @abstractmethod
    def load_model(self):
        """Load model from model path."""

    @abstractmethod
    def predict(self, target):
        """Return self model's raw output."""
