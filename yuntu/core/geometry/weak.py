import yuntu.core.utils.atlas as utils
from yuntu.core.geometry.base import Geometry


class Weak(Geometry):
    name = Geometry.Types.Weak

    def __init__(self, geometry=None):
        if geometry is None:
            geometry = utils.bbox_to_polygon([
                0, utils.INFINITY,
                0, utils.INFINITY
            ])
        super().__init__(geometry=geometry)

    def buffer(self, buffer=None, **kwargs):
        return self

    def shift(self, shift=None, **kwargs):
        return self

    def scale(self, scale=None, center=None, **kwargs):
        return self

    def transform(self, transform):
        return self

    def plot(self, ax=None, **kwargs):
        return super().plot(ax=ax, **kwargs)

    @property
    def bounds(self):
        return None, None, None, None
