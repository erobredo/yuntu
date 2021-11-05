"""Dataframe accesors for activity methods"""
import numpy as np
import pandas as pd
import pytz
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from yuntu.analytics.ecology.basic import diversity, richness, rarefaction, rarefaction_curve

ABS_START_TIME = 'abs_start_time'
ABS_END_TIME = 'abs_end_time'
REQUIRED_ACTIVITY_COLUMNS = [ABS_START_TIME,
                             ABS_END_TIME]

@pd.api.extensions.register_dataframe_accessor("activity")
class ActivityAccessor:
    def __init__(self, pandas_obj):
        abs_start_time_column = ABS_START_TIME
        abs_end_time_column = ABS_END_TIME

        self.activity_columns = None
        self._validate(self, pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(self, obj):
        for col in REQUIRED_ACTIVITY_COLUMNS:
            if col not in obj.columns:
                message = f"Not an activity dataframe. Missing '{col}' column."
                raise ValueError(message)

        self.activity_columns = []
        for col, dtype in zip(list(obj.columns), list(obj.dtypes)):
            if pd.api.types.is_float_dtype(dtype) and col not in REQUIRED_ACTIVITY_COLUMNS:
                self.activity_columns.append(col)
        self.activity_columns = list(set(self.activity_columns))

        if len(self.activity_columns) == 0:
            message = "Could not find any column to treat as activity."
            raise ValueError(message)

    def diversity(self, div_type="Shannon", component="alpha", labels=None):
        if component == "gamma":
            return diversity(self._obj.sum().values, div_type=div_type)

        if labels is None:
            labels = self.activity_columns

        div = self._obj.apply(diversity, labels=labels, div_type=div_type, axis=1)

        if component == "alpha":
            return div

        gamma = diversity(self._obj[labels].sum(), labels=labels, div_type=div_type)

        if component == "beta":
            return div.diversity.mean() / gamma

        if component == "partial_beta":
            div.loc[:, "diversity"] = div.diversity.apply(lambda x: x / gamma)
            return div

    def richness(self, labels=None, total=False):
        if labels is None:
            labels = self.activity_columns
        rich = self._obj.apply(richness, labels=labels, axis=1)
        if not total:
            return rich
        return rich.richness.sum()

    def rarefaction(self, size=None, labels=None):
        if labels is None:
            labels = self.activity_columns

        if size is None:
            sums = self._obj[labels].apply(sum, axis=1)
            size = np.min(sums)
            rarefact = self._obj.apply(rarefaction, size=size, labels=labels, axis=1)
        else:
            if isinstance(size, (int, float)):
                rarefact = self._obj.apply(rarefaction, size=size, labels=labels, axis=1)
            else:
                if len(size) != len(self.activity_columns):
                    msg = 'Size length should be equal to the number of labels.'
                    raise ValueError(msg)
                z = self._obj.copy()
                z['size'] = size
                s2 = z['size']
                x2 = z.drop('size')
                rarefact = x2.apply(rarefaction, size=s2, labels=labels, axis=1)

        return rarefact

    def plot(self, ax=None, time_format = '%d-%m-%Y %H:%M:%S', labels=None,
             view_time_zone="America/Mexico_city", nticks=15, stacked=True):

        if ax is None:
            ax = plt.gca()

        min_t = self._obj.abs_start_time.min().astimezone(view_time_zone)
        max_t = self._obj.abs_end_time.max().astimezone(view_time_zone)

        if labels is None:
            labels = self.activity_columns

        self._obj[labels].plot.area(ax=ax, stacked=stacked)

        nframes = self._obj.shape[0]
        step = float(nframes) / nticks

        total_time = datetime.timedelta.total_seconds(max_t - min_t)
        time_step = total_time / nticks

        xticks = [x for x in np.arange(0, nframes, step)]

        ax.set_xticks(xticks)
        xticklabels = [(min_t+datetime.timedelta(seconds=i*time_step)).strftime(format=time_format)
                       for i in range(len(xticks))]
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_ylabel("Detections")

        return ax

    def plot_diversity(self, ax=None, time_format = '%d-%m-%Y %H:%M:%S', labels=None,
                       view_time_zone="America/Mexico_city", nticks=15,
                       div_type="Shannon", component="alpha"):
        if ax is None:
            ax = plt.gca()

        min_t = self._obj.abs_start_time.min().astimezone(view_time_zone)
        max_t = self._obj.abs_end_time.max().astimezone(view_time_zone)

        div = self.diversity(div_type=div_type, component=component, labels=labels)

        if component in ["beta", "gamma"]:
            div.diversity.plot(ax=ax)
        else:
            div.diversity.plot.area(ax=ax)

            nframes = div.shape[0]

            step = float(nframes) / nticks

            total_time = datetime.timedelta.total_seconds(max_t - min_t)
            time_step = total_time / nticks

            xticks = [x for x in np.arange(0, nframes, step)]

            ax.set_xticks(xticks)
            xticklabels = [(min_t+datetime.timedelta(seconds=i*time_step)).strftime(format=time_format)
                           for i in range(len(xticks))]

            ax.set_xticklabels(xticklabels, rotation=90)

        if div_type == "Hill":
            ax.set_ylabel("Eff. number of species")
        elif component in ["partial_beta", "beta"]:
            ax.set_ylabel("Beta diversity")
        else:
            ax.set_ylabel("Diversity")

        return ax

    def plot_richness(self, ax=None, time_format = '%d-%m-%Y %H:%M:%S', labels=None,
                      view_time_zone="America/Mexico_city", total=False, nticks=15):
        if ax is None:
            ax = plt.gca()

        min_t = self._obj.abs_start_time.min().astimezone(view_time_zone)
        max_t = self._obj.abs_end_time.max().astimezone(view_time_zone)

        rich = self.richness(labels=labels, total=total)

        if total:
            rich.richness.plot(ax=ax)
        else:
            rich.richness.plot.area(ax=ax)

            nframes = rich.shape[0]
            step = float(nframes) / nticks
            total_time = datetime.timedelta.total_seconds(max_t - min_t)
            time_step = total_time / nticks
            xticks = [x for x in np.arange(0, nframes, step)]
            ax.set_xticks(xticks)
            xticklabels = [(min_t+datetime.timedelta(seconds=i*time_step)).strftime(format=time_format)
                           for i in range(len(xticks))]
            ax.set_xticklabels(xticklabels, rotation=90)

        ax.set_ylabel("Species count")

        return ax

    def rarefaction_curve(self, view_time_zone="America/Mexico_city",
                          include_total=True, labels=None, cmap=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if labels is None:
            labels = self.activity_columns

        z = self._obj.copy()
        z.reset_index(inplace=True)

        if cmap is None:
            cmap = cm.get_cmap('Spectral')

        if include_total:
            total_counts = (z[labels]
                            .sum()
                            .to_frame()
                            .transpose())
            total_counts.reset_index(inplace=True)
            total_counts.apply(rarefaction_curve, color="green", labels=labels,
                               plot_label="Total", exact=True, ax=ax, axis=1)

        z["color"] = np.array([x for x in np.arange(0,1,1/z.shape[0])])
        z.apply(rarefaction_curve, view_time_zone=view_time_zone, labels=labels, cmap=cmap, ax=ax, axis=1)

        ax.set_xlabel('Number of detections')
        ax.set_ylabel('Number of species')

        return ax
