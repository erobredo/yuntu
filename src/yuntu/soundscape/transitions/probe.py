import gc
import os
import json
import shutil
import numpy as np
from pony.orm import db_session
import dask.bag

from yuntu.utils import module_object
from yuntu import Audio
from yuntu.collection.methods import collection
from yuntu.core.pipeline.places import *
from yuntu.core.pipeline import transition

def write_probe_outputs(partition, probe_config, col_config, write_config, batch_size, overwrite=False):
    """Run probe on partition and write results"""

    col = collection(**col_config)

    with db_session:
        dataframe = col.get_recording_dataframe(query=partition["query"],
                                                offset=partition["offset"],
                                                limit=partition["limit"],
                                                with_annotations=True,
                                                with_metadata=True)
    col.db_manager.db.disconnect()

    probe_class = module_object(probe_config["module"])
    probe_kwargs = probe_config["kwargs"]

    rows = []
    count = 0
    with probe_class(**probe_kwargs) as probe:
        for rid, path, duration, timeexp in dataframe[["id", "path", "duration", "timeexp"]].values:
            new_row = {"id": rid}
            out_path = os.path.join(write_config["write_dir"], f"{rid}.npy")

            if os.path.exists(out_path) and not overwrite:
                raw_output = np.load(out_path)
            else:
                with Audio(path=path, timeexp=timeexp) as audio:
                    raw_output = probe.predict(audio, batch_size)
                    np.save(out_path, raw_output)

            new_row["probe_output"] = out_path

            count += 1

            if count % 10 == 0:
                gc.collect()

            rows.append(new_row)

    return rows

def probe_all(recording_rows, probe_config):
    """Run probe row by row"""
    probe_class = module_object(probe_config["module"])
    probe_kwargs = probe_config["kwargs"]

    if "annotate_args" in probe_config:
        annotate_args = probe_config["annotate_args"]
    else:
        annotate_args = {}

    all_annotations = []
    with probe_class(**probe_kwargs) as probe:
        for row in recording_rows:
            rid = row["id"]
            path = row["path"]
            duration = row["duration"]
            timeexp = row["timeexp"]
            samplerate = row["samplerate"]

            prev_annotations = []
            if "annotations" in row:
                prev_annotations = row["annotations"]

            if "meta_arg_extractor" in probe_config:
                extractor_config = probe_config["meta_arg_extractor"]
                if "kwargs" in extractor_config:
                    arg_extractor = module_object(extractor_config["module"])(**extractor_config["kwargs"])
                else:
                    arg_extractor = module_object(extractor_config["module"])

                annotate_args.update(arg_extractor(row))

            with Audio(path=path, duration=duration, samplerate=samplerate, timeexp=timeexp, annotations=prev_annotations) as audio:
                annotations = probe.annotate(audio, **annotate_args)

            for i in range(len(annotations)):
                annotations[i]["recording"] = rid
                annotations[i]["labels"] = json.dumps(annotations[i]["labels"])
                annotations[i]["metadata"] = json.dumps(annotations[i]["metadata"])
                all_annotations.append(annotations[i])


    return all_annotations

def insert_probe_annotations(partition, probe_config, col_config, overwrite=False):
    """Run probe on partition and write results"""

    col = collection(**col_config)

    use_annotations = True
    use_metadata = True

    if "use_annotations" in probe_config:
        use_annotations = probe_config["use_annotations"]
    if "use_metadata" in probe_config:
        use_metadata = probe_config["use_metadata"]

    with db_session:
        dataframe = col.get_recording_dataframe(query=partition["query"],
                                                offset=partition["offset"],
                                                limit=partition["limit"],
                                                with_annotations=use_annotations,
                                                with_metadata=use_metadata)

    probe_class = module_object(probe_config["module"])
    probe_kwargs = probe_config["kwargs"]

    if "annotate_args" in probe_config:
        annotate_args = probe_config["annotate_args"]
    else:
        annotate_args = {}

    rows = []
    count = 0
    with probe_class(**probe_kwargs) as probe:
        for row in dataframe.to_dict(orient="records"):
            rid = row["id"]
            path = row["path"]
            duration = row["duration"]
            timeexp = row["timeexp"]
            samplerate = row["samplerate"]

            prev_annotations = []
            if "annotations" in row:
                prev_annotations = row["annotations"]

            if "meta_arg_extractor" in probe_config:
                extractor_config = probe_config["meta_arg_extractor"]
                if "kwargs" in extractor_config:
                    arg_extractor = module_object(extractor_config["module"])(**extractor_config["kwargs"])
                else:
                    arg_extractor = module_object(extractor_config["module"])
                annotate_args.update(arg_extractor(row))

            with Audio(path=path, duration=duration, samplerate=samplerate, timeexp=timeexp, annotations=prev_annotations) as audio:
                annotations = probe.annotate(audio, **annotate_args)

            if len(annotations) > 0:
                with db_session:
                    recording = col.recordings(query=lambda recording: recording.id == rid, iterate=False)[0]
                    for i in range(len(annotations)):
                        annotations[i]["recording"] = recording
                    col.annotate(annotations)

            new_row = {"id": rid}
            new_row["annotation_inserts"] = len(annotations)

            count += 1

            if count % 10 == 0:
                gc.collect()

            rows.append(new_row)

    col.db_manager.db.disconnect()

    return rows


@transition(name="probe_write", outputs=["write_result"], persist=True,
            signature=((DynamicPlace, DictPlace, DictPlace, DictPlace, ScalarPlace, BoolPlace), (DaskDataFramePlace,)))
def probe_write(partitions, probe_config, col_config, write_config, batch_size, overwrite=False):
    """Run probe and write results for each partition in parallel"""

    if os.path.exists(write_config["write_dir"]) and overwrite:
        shutil.rmtree(write_config["write_dir"], ignore_errors=True)
    if not os.path.exists(write_config["write_dir"]):
        os.makedirs(write_config["write_dir"])

    results = partitions.map(write_probe_outputs,
                             probe_config=probe_config,
                             col_config=col_config,
                             write_config=write_config,
                             batch_size=batch_size).flatten()

    meta = [('id', np.dtype('int')), ('probe_output', np.dtype('<U'))]

    return results.to_dataframe(meta=meta)


@transition(name="probe_annotate", outputs=["annotation_result"], persist=True,
            signature=((DynamicPlace, DictPlace, DictPlace), (DaskDataFramePlace,)))
def probe_annotate(partitions, probe_config, col_config):
    """Run probe and annotate recordings for each partition in parallel"""

    results = partitions.map(insert_probe_annotations,
                             probe_config=probe_config,
                             col_config=col_config).flatten()

    meta = [('id', np.dtype('int')), ('annotation_inserts', np.dtype('int'))]

    return results.to_dataframe(meta=meta)


@transition(name="probe_recordings", outputs=["matches"], persist=True,
            signature=((DynamicPlace, DictPlace), (DaskDataFramePlace,)))
def probe_recordings(recordings_bag, probe_config):
    """Run probe and annotate bag of recording rows."""

    meta = [('recording', np.dtype('int')),
            ('start_time', np.dtype('float64')),
            ('end_time', np.dtype('float64')),
            ('min_freq', np.dtype('float64')),
            ('max_freq', np.dtype('float64')),
            ('type', np.dtype('O')),
            ('metadata', np.dtype('O')),
            ('geometry', np.dtype('O')),
            ('labels', np.dtype('O'))]

    results = recordings_bag.map(probe_all,
                                 probe_config=probe_config).flatten()

    return results.to_dataframe(meta=meta)
