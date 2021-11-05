"""Distinct types of annotations."""
from pony.orm import Required
from pony.orm import Optional
from pony.orm import PrimaryKey
from pony.orm import Json
from datetime import datetime


WEAK_ANNOTATION = 'WeakAnnotation'
TIME_INTERVAL_ANNOTATION = 'TimeIntervalAnnotation'
FREQUENCY_INTERVAL_ANNOTATION = 'FrequencyIntervalAnnotation'
BBOX_ANNOTATION = 'BBoxAnnotation'
LINESTRING_ANNOTATION = 'LineStringAnnotation'
POLYGON_ANNOTATION = 'PolygonAnnotation'


ANNOTATION_TYPES = [
    WEAK_ANNOTATION,
    TIME_INTERVAL_ANNOTATION,
    FREQUENCY_INTERVAL_ANNOTATION,
    BBOX_ANNOTATION,
    LINESTRING_ANNOTATION,
    POLYGON_ANNOTATION,
]


def build_base_annotation_model(db):
    """Create base annotation model."""
    class Annotation(db.Entity):
        """Basic annotation entity for yuntu."""

        id = PrimaryKey(int, auto=True)
        recording = Required('Recording')

        type = Required(str)
        labels = Required(Json)
        metadata = Required(Json)

        start_time = Optional(float)
        end_time = Optional(float)
        max_freq = Optional(float)
        min_freq = Optional(float)

        geometry = Required(str)

        def before_insert(self):
            if self.type not in ANNOTATION_TYPES:
                message = f'Notetype {self.type} not implemented'
                raise NotImplementedError(message)

            if self.type == WEAK_ANNOTATION:
                return

            if self.type is None or self.end_time is None:
                message = (
                    f'Annotation type {self.type} requires setting '
                    'a starting and ending time (start_time and end_time)')
                raise ValueError(message)

            if self.type == TIME_INTERVAL_ANNOTATION:
                return

            if self.max_freq is None or self.min_freq is None:
                message = (
                    f'Annotation type {self.type} requires setting '
                    'a maximum and minimum frequency (max_freq and min_freq)')
                raise ValueError(message)

            if self.type == BBOX_ANNOTATION:
                return

            if self.geometry is None:
                message = (
                    f'Annotation type {self.type} requires setting '
                    'a geometry string (in wkt format).')
                raise ValueError(message)
    return Annotation

def build_timed_annotation_model(Annotation):
    class TimedAnnotation(Annotation):
        """Datastore that builds data from a foreign database."""
        abs_start_time = Optional(datetime, precision=6)
        abs_end_time = Optional(datetime, precision=6)

    return TimedAnnotation
