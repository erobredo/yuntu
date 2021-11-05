import numpy as np

from yuntu.core.media.base import Media


class MaskedMediaMixin:
    def __init__(self, media, geometry, **kwargs):
        self.geometry = geometry
        self.media = media
        kwargs['window'] = self.media.window
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            'geometry': self.geometry.to_dict(),
            'media': self.media.to_dict(),
            **super().to_dict()
        }

    def _copy_dict(self):
        return {
            'geometry': self.geometry,
            'media': self.media,
            **super()._copy_dict(),
        }

    def plot(self, ax=None, **kwargs):
        ax = super().plot(ax=ax, **kwargs)

        if kwargs.get('media', False):
            media_kwargs = kwargs.get('media_kwargs', {})
            media = self.media.copy()
            media._array = np.ma.masked_array(
                media.array,
                mask=1 - self.array)
            ax = media.plot(ax=ax, **media_kwargs)

        return ax

    def union(self, other, lazy=False):
        assert isinstance(other, MaskedMediaMixin)
        geometry = self.geometry.union(other.geometry)
        kwargs = self._copy_dict()
        kwargs['geometry'] = geometry

        if not self.is_empty() and not other.is_empty() and not lazy:
            array = self.array | other.array
            kwargs['array'] = array

        return type(self)(**kwargs, lazy=lazy)

    def intersection(self, other, lazy=False):
        assert isinstance(other, MaskedMediaMixin)
        geometry = self.geometry.intersection(other.geometry)
        kwargs = self._copy_dict()
        kwargs['geometry'] = geometry

        if not self.is_empty() and not other.is_empty() and not lazy:
            array = self.array & other.array
            kwargs['array'] = array

        return type(self)(**kwargs, lazy=lazy)

    def __or__(self, other):
        if not isinstance(other, MaskedMediaMixin):
            return super().__or__(other)

        return self.union(other, lazy=False)

    def __and__(self, other):
        if not isinstance(other, MaskedMediaMixin):
            return super().__and__(other)

        return self.intersection(other, lazy=False)

    def write(self, path=None, **kwargs):
        if path is None:
            path = self.path

        self.path = path

        data = {
            'mask': self.array
        }

        if self.media.path_exists():
            data['media_path'] = self.media.path

        if not self.window.is_trivial():
            window_data = {
                f'window_{key}': value
                for key, value in self.window.to_dict()
                if value is not None
            }
            data.update(window_data)

        np.savez(self.path, **data)


class MaskedMedia(MaskedMediaMixin, Media):
    pass


def masks(cls):
    def decorator(mask_cls):
        if not issubclass(mask_cls, MaskedMediaMixin):
            message = (
                f'Class {mask_cls} cannot be masked by an'
                ' object which does not inherit from MaskedMedia')
            raise ValueError(message)

        cls.mask_class = mask_cls
        return mask_cls
    return decorator
