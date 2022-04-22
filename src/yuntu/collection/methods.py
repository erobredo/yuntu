import os
import json
from yuntu.core.audio.utils import media_open
from yuntu.collection.base import Collection, TimedCollection, SpatialCollection, SpatioTemporalCollection

def collection(col_type="simple", materialized=None, **kwargs):
    if materialized is not None:
        return load_materialized(materialized)
    if col_type == "simple":
        return Collection(**kwargs)
    elif col_type == "timed":
        return TimedCollection(**kwargs)
    elif col_type == "spatial":
        return SpatialCollection(**kwargs)
    elif col_type == "spatiotemporal":
        return SpatioTemporalCollection(**kwargs)
    raise NotImplementedError(f"Collection type {col_type} unknown")

def load_materialized(materialized):
    col_config_path = os.path.join(materialized, "col_config.json")
    with media_open(col_config_path) as f:
        col_config = json.load(f)
    col_config["base_path"] = materialized
    return collection(**col_config)
