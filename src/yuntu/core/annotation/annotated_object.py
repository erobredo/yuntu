"""
Annotated Object Module.

This module defines a Mixin that can be given to all
objects that posses annotations.
"""
from inspect import signature
import pandas as pd
from yuntu.core.annotation.annotation import Annotation


class AnnotationList(list):
    def to_dict(self):
        return [
            annotation.to_dict() for annotation in self
        ]

    def add(
            self,
            annotation=None,
            geometry=None,
            labels=None,
            metadata=None,
            id=None):
        """Append annotation to AnnotationList."""
        if annotation is None:
            annotation = Annotation(
                geometry=geometry,
                labels=labels,
                metadata=metadata,
                id=id)

        if not isinstance(annotation, Annotation):
            annotation = Annotation.from_dict(annotation)

        self.append(annotation)

    def add_multiple(self, annotations):
        """Append annotations to AnnotationList."""
        for annotation in annotations:
            if not isinstance(annotation, Annotation):
                annotation = Annotation.from_dict(annotation)

            self.append(annotation)

    def to_dataframe(self):
        """Produce pandas DataFrame from AnnotationList."""
        data = []
        for annotation in self:
            row = {
                'id': annotation.id,
                'type': type(annotation).__name__,
                'start_time': annotation.geometry.bounds[0],
                'end_time': annotation.geometry.bounds[2],
                'min_freq': annotation.geometry.bounds[1],
                'max_freq': annotation.geometry.bounds[3]
            }
            for label in annotation.iter_labels():
                row[label.key] = label.value
            row['geometry'] = annotation.geometry
            data.append(row)
        return pd.DataFrame(data)

    def plot(self, ax=None, **kwargs):
        """Plot all annotations."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (15, 5)))

        key = kwargs.get('key', None)
        for annotation in self:
            if not annotation.has_label(key, mode=kwargs.get('filter', 'all')):
                continue
            annotation.plot(ax=ax, **kwargs)

        if kwargs.get('legend', False):
            ax.legend()

        return ax

    def buffer(self, buffer=None, **kwargs):
        annotations = [
            annotation.buffer(buffer=buffer, **kwargs)
            for annotation in self]
        return AnnotationList(annotations)

    def apply(self, func):
        annotations = [
            func(annotation) for annotation
            in self]
        return AnnotationList(annotations)

    def filter(self, func):
        """Return new AnnotationList with filtered annotations."""
        annotations = [
            annotation for annotation in self
            if func(annotation)]
        return AnnotationList(annotations)


class AnnotatedObjectMixin:
    def __init__(
            self,
            annotations=None,
            filter_annotations=True,
            **kwargs):
        if annotations is None:
            annotations = []

        annotations = self._cast_annotations(annotations)

        if filter_annotations:
            annotations = self._filter_annotations(annotations)

        self.annotations = AnnotationList(annotations)
#         if len([key for key in signature(super().__init__).parameters.keys() if key != "self"]) != 0:
#             super().__init__(**kwargs)
#         else:
#             super().__init__()
        super().__init__()

    def to_dict(self):
        return {
            'annotations': self.annotations.to_dict()
        }

    @staticmethod
    def _cast_annotations(annotations):
        new_annotations = []
        for annotation in annotations:
            if not isinstance(annotation, Annotation):
                annotation = Annotation.from_dict(annotation)

            new_annotations.append(annotation)

        return new_annotations

    def _filter_annotations(self, annotation_list):
        if not hasattr(self, 'window'):
            return annotation_list

        if self.window is None:
            return annotation_list

        if self.window.is_trivial():
            return annotation_list

        filtered = [
            annotation for annotation in annotation_list
            if annotation.intersects(self.window)
        ]

        return filtered

    def annotate(
            self,
            annotation=None,
            geometry=None,
            labels=None,
            metadata=None,
            id=None):
        self.annotations.add(
            annotation=annotation,
            geometry=geometry,
            labels=labels,
            metadata=metadata,
            id=id)

    def add_annotations(self, annotations):
        self.annotations.add_multiple(annotations)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', None))

        if kwargs.get('annotations', False):
            annotations_kwargs = kwargs.get('annotation_kwargs', {})
            ax = self.annotations.plot(ax=ax, **annotations_kwargs)

        return ax
