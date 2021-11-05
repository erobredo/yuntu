from yuntu.core.media.base import Media


class HistogramMediaMixin:
    histogram_axis = -1

    def __init__(self, bins, range, histogram_axis=None, **kwargs):
        self.bins = bins
        self.range = range

        if histogram_axis is not None:
            self.histogram_axis = histogram_axis

        super().__init__(**kwargs)

    def _get_axis_info(self):
        return {
            'histogram_axis': self.histogram_axis,
            'bins': self.bins.tolist(),
            **super()._get_axis_info()
        }


class HistogramMedia(HistogramMediaMixin, Media):
    pass
