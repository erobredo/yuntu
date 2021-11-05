"""Base classes for reference systems.

An atlas is a collection of charts with a reference system over time and
frequency.
"""
from yuntu.core.utils.atlas import bbox_to_polygon, \
                                   plot_geometry, \
                                   reference_system, \
                                   build_multigeometry


class Chart:
    """A delimited region of time and frequency."""

    def __init__(self, start_time, end_time, min_freq, max_freq):
        """Initialize chart."""
        self._bbox = None
        self._wkt = None
        self._geometry = None
        self.start_time = start_time
        self.end_time = end_time
        self.min_freq = min_freq
        self.max_freq = max_freq

    def __repr__(self):
        """Repr chart."""
        return f'Chart: ({self.wkt})'

    def __str__(self):
        """Chart as string."""
        return self.wkt

    def to_dict(self):
        """Chart to dict."""
        return {"start_time": self.start_time,
                "end_time": self.end_time,
                "min_freq": self.min_freq,
                "max_freq": self.max_freq}

    @property
    def bbox(self):
        """Return bounding box of chart."""
        if self._bbox is None:
            self._bbox = (self.start_time, self.end_time,
                          self.min_freq, self.max_freq)
        return self._bbox

    @property
    def geometry(self):
        """Return chart as shapely Polygon."""
        if self._geometry is None:
            self._geometry = bbox_to_polygon(self.bbox)
        return self._geometry

    @property
    def wkt(self):
        """Return chart geometry as wkt."""
        if self._wkt is None:
            self._wkt = self.geometry.wkt
        return self._wkt

    def plot(self, ax=None, outpath=None, **kwargs):
        """Plot chart geometry."""
        plot_geometry(self.geometry, ax, outpath, **kwargs)


class Atlas:
    """A collection of compatible charts within boundaries."""

    def __init__(self,
                 time_win,
                 time_hop,
                 freq_win,
                 freq_hop,
                 bounds,
                 center=None):
        if time_win > bounds[1] - bounds[0] or \
           freq_win > bounds[3] - bounds[2]:
            raise ValueError("Window larger than bounds.")
        if center is not None:
            if center[0] < bounds[0] or \
               center[1] < bounds[2] or \
               center[0] > bounds[1] or \
               center[1] > bounds[3]:
                raise ValueError("Center outside bounds.")
        self._bbox = None
        self._geometry = None
        self._atlas = {}
        self.shape = None
        self.xrange = None
        self.yrange = None
        self.time_win = time_win
        self.time_hop = time_hop
        self.freq_win = freq_win
        self.freq_hop = freq_hop
        self.bounds = bounds
        self.center = center
        self.build()

    def __iter__(self):
        """Iterate over charts within atlas."""
        for coords in self._atlas:
            yield self._atlas[coords], coords

    def __and__(self, other):
        """Return atlas intersection.

        Returns new atlas where bounds are the intersection of self and other
        bounds and windows and hops are minimal.
        """
        start_time = max(self.bounds[0], other.bounds[0])
        end_time = min(self.bounds[1], other.bounds[1])
        min_freq = max(self.bounds[2], other.bounds[2])
        max_freq = min(self.bounds[3], other.bounds[3])

        if end_time - start_time <= 0 or max_freq - min_freq <= 0:
            return None

        time_win = min(self.time_win, other.time_win)
        freq_win = min(self.freq_win, other.freq_win)
        time_hop = min(self.time_hop, other.time_hop)
        freq_hop = min(self.freq_hop, other.freq_hop)

        return Atlas(time_win, time_hop, freq_win, freq_hop,
                     (start_time, end_time, min_freq, max_freq))

    def __or__(self, other):
        """Return atlas union.

        Returns new atlas where bounds are the union of self and other
        bounds and windows and hops are maximal.
        """
        start_time = min(self.bounds[0], other.bounds[0])
        end_time = max(self.bounds[1], other.bounds[1])
        min_freq = min(self.bounds[2], other.bounds[2])
        max_freq = max(self.bounds[3], other.bounds[3])

        if end_time - start_time <= 0 or max_freq - min_freq <= 0:
            return None

        time_win = max(self.time_win, other.time_win)
        freq_win = max(self.freq_win, other.freq_win)
        time_hop = max(self.time_hop, other.time_hop)
        freq_hop = max(self.freq_hop, other.freq_hop)

        return Atlas(time_win, time_hop, freq_win, freq_hop,
                     (start_time, end_time, min_freq, max_freq))

    def build(self):
        """Build system of charts based on input parameters."""
        ref_system, self.shape, \
            self.xrange, self.yrange = reference_system(self.time_win,
                                                        self.time_hop,
                                                        self.freq_win,
                                                        self.freq_hop,
                                                        self.bounds,
                                                        self.center)
        for coords in ref_system:
            self._atlas[coords] = Chart(*ref_system[coords])

    def chart(self, atlas_coords):
        """Return chart at specified atlas coordinates.

        Parameters
        ----------
        atlas_coords: tuple(int, int)
            The corresponding coordintaes in the reference system.

        Returns
        -------
            Chart at coordinates.
        """

        if atlas_coords in self._atlas:
            return self._atlas[atlas_coords]
        raise ValueError("Atlas coordinates out of range.")

    def intersects(self, geometry):
        """Return charts that intersect geometry.

        Parameters
        ----------
        geometry: shapely.geometry
            Geometry to operate.

        Returns
        -------
        charts: list
            List of matching charts.
        """
        return [(self._atlas[coords], coords)
                for coords in self._atlas
                if self._atlas[coords].geometry.intersects(geometry)]

    def within(self, geometry):
        """Return charts that lie within geometry.

        Parameters
        ----------
        geometry: shapely.geometry
            Geometry to operate.

        Returns
        -------
        charts: list
            List of matching charts.
        """
        return [(self._atlas[coords], coords)
                for coords in self._atlas
                if self._atlas[coords].geometry.within(geometry)]

    def contains(self, geometry):
        """Return charts that contain geometry.

        Parameters
        ----------
        geometry: shapely.geometry
            Geometry to operate.

        Returns
        -------
        charts: list
            List of matching charts.
        """
        return [(self._atlas[coords], coords)
                for coords in self._atlas
                if self._atlas[coords].geometry.contains(geometry)]

    @property
    def bbox(self):
        """Return bounding box of atlas."""
        if self._bbox is None:
            self._bbox = bbox_to_polygon(self.bounds)
        return self._bbox

    @property
    def geometry(self):
        """Return atlas geometry as MultiPolygon."""
        if self._geometry is None:
            geom_arr = [self._atlas[coords].geometry
                        for coords in self._atlas]
            self._geometry = build_multigeometry(geom_arr, "Polygon")
        return self._geometry

    def plot(self, ax=None, outpath=None, **kwargs):
        """Plot atlas."""
        all_coords = list(self._atlas.keys())
        ncharts = len(all_coords)
        for i in range(ncharts - 1):
            plot_geometry(self._atlas[all_coords[i]].geometry, ax, **kwargs)
        plot_geometry(self._atlas[all_coords[ncharts - 1]].geometry,
                      ax=ax, outpath=outpath, **kwargs)
