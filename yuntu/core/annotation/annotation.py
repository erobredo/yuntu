from uuid import uuid4
from abc import ABC
from enum import Enum

from yuntu.core.annotation.labels import Labels
import yuntu.core.geometry as geom


class Annotation(ABC):
    name = None
    geometry_class = None

    class Types(Enum):
        WEAK = 'WeakAnnotation'
        TIME_LINE = 'TimeLineAnnotation'
        TIME_INTERVAL = 'TimeIntervalAnnotation'
        FREQUENCY_LINE = 'FrequencyLine'
        FREQUENCY_INTERVAL = 'FrequencyIntervalAnnotation'
        BBOX = 'BBoxAnnotation'
        LINESTRING = 'LineStringAnnotation'
        POLYGON = 'PolygonAnnotation'
        POINT = 'Point'

    # pylint: disable=redefined-builtin
    def __init__(
            self,
            labels=None,
            id=None,
            metadata=None,
            geometry=None):

        if not isinstance(geometry, self.geometry_class):
            message = (
                'The given geometry is not of the correct type. '
                f'Should be {self.geometry_class} but was given '
                f'{type(geometry)}.s')
            raise ValueError(message)

        if id is None:
            id = str(uuid4())
        self.id = id

        if labels is None:
            labels = Labels([])

        if not isinstance(labels, Labels):
            labels = Labels(labels)

        self.labels = labels
        self.metadata = metadata
        self.geometry = geometry

    def __repr__(self):
        args = {
            'labels': repr(self.labels),
            'geometry': repr(self.geometry)
        }

        args_string = ', '.join([
            '{}={}'.format(key, value)
            for key, value in args.items()
        ])
        name = type(self).__name__
        return f'{name}({args_string})'

    def has_label(self, key, mode='all'):
        if key is None:
            return True

        if not isinstance(key, (tuple, list)):
            return key in self.labels

        if mode == 'all':
            for subkey in key:
                if subkey not in self.labels:
                    return False
            return True

        if mode == 'any':
            for subkey in key:
                if subkey in self.labels:
                    return True
            return False

        message = 'Mode must be "all" or "any"'
        raise ValueError(message)

    @property
    def type(self):
        return self.name

    def iter_labels(self):
        return iter(self.labels)

    def get_label(self, key):
        return self.labels.get(key)

    def get_label_value(self, key):
        return self.labels.get_value(key)

    def get_window(self):
        return self.geometry.window

    def _copy_dict(self):
        data = self.to_dict()
        data['labels'] = self.labels
        data['geometry'] = self.geometry
        data.pop('type')
        return data

    def _get_buffer_class(self):
        if isinstance(self, (
                WeakAnnotation,
                TimeIntervalAnnotation,
                BBoxAnnotation,
                FrequencyIntervalAnnotation)):
            return type(self)

        if isinstance(self, TimeLineAnnotation):
            return TimeIntervalAnnotation

        if isinstance(self, FrequencyLineAnnotation):
            return FrequencyIntervalAnnotation

        return PolygonAnnotation

    def buffer(self, buffer=None, **kwargs):
        data = self._copy_dict()
        data['geometry'] = self.geometry.buffer(buffer=buffer, **kwargs)
        return self._get_buffer_class()(**data)

    def shift(self, shift=None, **kwargs):
        data = self._copy_dict()
        data['geometry'] = self.geometry.shift(shift=shift, **kwargs)
        return self._get_buffer_class()(**data)

    def scale(self, scale=None, **kwargs):
        data = self._copy_dict()
        data['geometry'] = self.geometry.scale(scale=scale, **kwargs)
        return self._get_buffer_class()(**data)

    def transform(self, transform):
        data = self._copy_dict()
        data['geometry'] = self.geometry.transform(transform)
        return self._get_buffer_class()(**data)

    def intersects(self, other):
        if isinstance(other, Annotation):
            return self.geometry.intersects(other.geometry)

        return self.geometry.intersects(other)

    def add_label(
            self,
            value=None,
            key=None,
            type=None,
            data=None,
            label=None):
        self.labels.add(
            value=value,
            key=key,
            type=type,
            data=data,
            label=label)

    def to_dict(self):
        data = {
            'id': self.id,
            'labels': self.labels.to_dict(),
            'type': self.type.value,
            'geometry': self.geometry.to_dict()
        }

        if self.metadata is not None:
            data['metadata'] = self.metadata

        return data

    @staticmethod
    def _type_to_class(annotation_type):
        if annotation_type == Annotation.Types.WEAK.value:
            return WeakAnnotation

        if annotation_type == Annotation.Types.TIME_LINE.value:
            return TimeLineAnnotation

        if annotation_type == Annotation.Types.TIME_INTERVAL.value:
            return TimeIntervalAnnotation

        if annotation_type == Annotation.Types.FREQUENCY_LINE.value:
            return FrequencyLineAnnotation

        if annotation_type == Annotation.Types.FREQUENCY_INTERVAL.value:
            return FrequencyIntervalAnnotation

        if annotation_type == Annotation.Types.BBOX.value:
            return BBoxAnnotation

        if annotation_type == Annotation.Types.LINESTRING.value:
            return LineStringAnnotation

        if annotation_type == Annotation.Types.POLYGON.value:
            return PolygonAnnotation

        message = 'Annotation is of unknown type.'
        raise ValueError(message)

    @staticmethod
    def from_dict(data):
        data = data.copy()
        annotation_type = data.pop('type')
        annotation_class = Annotation._type_to_class(annotation_type)

        data['labels'] = Labels.from_dict(data['labels'])

        geometry_data = data['geometry']
        if 'type' not in geometry_data:
            geometry_data['type'] = annotation_class.geometry_class.name.name

        data['geometry'] = geom.Geometry.from_dict(geometry_data)

        datakeys = list(data.keys())
        for key in datakeys:
            if key not in ["labels", "id", "metadata", "geometry"]:
                del data[key]

        return annotation_class(**data)

    @staticmethod
    def from_record(record):
        data = record.copy()

        annotation_type = data['type']
        cls = Annotation._type_to_class(annotation_type)

        if 'geometry' in data:
            data['geometry'] = {
                'type': cls.geometry_class.name.value,
                'wkt': data['geometry']
            }

        return Annotation.from_dict(data)

    def plot(self, ax=None, **kwargs):
        ax = self.geometry.plot(ax=ax, **kwargs)

        if kwargs.get('label', False):
            ax = self.plot_labels(ax, **kwargs)

        return ax

    def plot_labels(self, ax, **kwargs):
        from matplotlib import transforms

        label = self.labels.get_text_repr(kwargs.get('key', None))
        left, bottom, right, top = self.geometry.bounds

        xcoord_position = kwargs.get('label_xposition', 'center')
        if left is None or right is None:
            xtransform = ax.transAxes

            if not isinstance(xcoord_position, (int, float)):
                xcoord = 0.5
        else:
            xtransform = ax.transData

            if isinstance(xcoord_position, str):
                if xcoord_position == 'left':
                    xcoord = left

                elif xcoord_position == 'right':
                    xcoord = right

                elif xcoord_position == 'center':
                    xcoord = (left + right) / 2

                else:
                    message = (
                        'label_xposition can only be a float, '
                        'or "left"/"right"/"center"')
                    raise ValueError(message)

            if isinstance(xcoord_position, float):
                xcoord = left + (right - left) * xcoord_position

        ycoord_position = kwargs.get('label_yposition', 'center')
        if bottom is None or top is None:
            ytransform = ax.transAxes

            if not isinstance(ycoord_position, (int, float)):
                ycoord = 0.5
        else:
            ytransform = ax.transData

            if isinstance(ycoord_position, str):
                if ycoord_position == 'top':
                    ycoord = top

                elif ycoord_position == 'bottom':
                    ycoord = bottom

                elif ycoord_position == 'center':
                    ycoord = (top + bottom) / 2

                else:
                    message = (
                        'ycoord_position can only be a float, '
                        'or "top"/"bottom"/"center"')
                    raise ValueError(message)

            if isinstance(ycoord_position, float):
                ycoord = bottom + (top - bottom) * ycoord_position

        trans = transforms.blended_transform_factory(xtransform, ytransform)
        ax.text(
            xcoord,
            ycoord,
            label,
            transform=trans,
            ha=kwargs.get('label_ha', 'center'))

        return ax

    def cut(self, other):
        return other.cut(self.get_window())

    def to_weak(self):
        return WeakAnnotation(
            labels=self.labels,
            id=self.id,
            metadata=self.metadata)


class WeakAnnotation(Annotation):
    name = Annotation.Types.WEAK
    geometry_class = geom.Weak

    def __init__(self, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.Weak()

        super().__init__(**kwargs)


class TimeLineAnnotation(Annotation):
    name = Annotation.Types.TIME_LINE
    geometry_class = geom.TimeLine

    def __init__(self, time=None, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.TimeLine(time=time)

        super().__init__(**kwargs)


class TimeIntervalAnnotationMixin:
    def to_start_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_start_line()
        return TimeLineAnnotation(**data)

    def to_end_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_end_line()
        return TimeLineAnnotation(**data)

    def to_time_center_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_time_center_line()
        return TimeLineAnnotation(**data)


class TimeIntervalAnnotation(TimeIntervalAnnotationMixin, Annotation):
    name = Annotation.Types.TIME_INTERVAL
    geometry_class = geom.TimeInterval

    def __init__(self, start_time=None, end_time=None, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.TimeInterval(
                start_time=start_time,
                end_time=end_time)

        super().__init__(**kwargs)


class FrequencyLineAnnotation(Annotation):
    name = Annotation.Types.FREQUENCY_LINE
    geometry_class = geom.FrequencyLine

    def __init__(self, freq=None, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.FrequencyLine(freq=freq)

        super().__init__(**kwargs)


class FrequencyIntervalAnnotationMixin:
    def to_min_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.min_line
        return FrequencyLineAnnotation(**data)

    def to_max_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_max_line()
        return FrequencyLineAnnotation(**data)

    def to_freq_center_line(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_freq_center_line()
        return FrequencyLineAnnotation(**data)


class FrequencyIntervalAnnotation(
        FrequencyIntervalAnnotationMixin,
        Annotation):
    name = Annotation.Types.FREQUENCY_INTERVAL
    geometry_class = geom.FrequencyInterval

    def __init__(self, min_freq=None, max_freq=None, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.FrequencyInterval(
                min_freq=min_freq,
                max_freq=max_freq)

        super().__init__(**kwargs)


class PointAnnotation(Annotation):
    name = Annotation.Types.POINT
    geometry_class = geom.Point

    def __init__(self, time=None, freq=None, **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.Point(time=time, freq=freq)

        super().__init__(**kwargs)


class Annotation2DMixin(
        TimeIntervalAnnotationMixin,
        FrequencyIntervalAnnotationMixin):
    def to_bbox(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_bbox()
        return BBoxAnnotation(**data)

    def to_center(self):
        data = self._copy_dict()
        data['geometry'] = self.geometry.to_center()
        return PointAnnotation(**data)


class BBoxAnnotation(Annotation2DMixin, Annotation):
    name = Annotation.Types.BBOX
    geometry_class = geom.BBox

    def __init__(
            self,
            start_time=None,
            end_time=None,
            min_freq=None,
            max_freq=None,
            **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.BBox(
                start_time=start_time,
                end_time=end_time,
                min_freq=min_freq,
                max_freq=max_freq)

        super().__init__(**kwargs)


class LineStringAnnotation(Annotation2DMixin, Annotation):
    name = Annotation.Types.LINESTRING
    geometry_class = geom.LineString

    def __init__(
            self,
            vertices=None,
            wkt=None,
            **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.LineString(
                wkt=wkt,
                vertices=vertices)

        super().__init__(**kwargs)


class PolygonAnnotation(Annotation2DMixin, Annotation):
    name = Annotation.Types.POLYGON
    geometry_class = geom.Polygon

    def __init__(
            self,
            wkt=None,
            shell=None,
            holes=None,
            **kwargs):
        if 'geometry' not in kwargs:
            kwargs['geometry'] = geom.Polygon(
                wkt=wkt,
                shell=shell,
                holes=holes)

        super().__init__(**kwargs)
