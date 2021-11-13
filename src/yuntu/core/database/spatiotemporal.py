'''Database manager with both time and spatial capabilities'''
from yuntu.core.database.spatial import SpatialDatabaseManager
from yuntu.core.database.annotations import build_timed_annotation_model
from yuntu.core.database.recordings import build_spatio_temporal_recording_model


class SpatioTemporalDatabaseManager(SpatialDatabaseManager):

    def build_spatialized_recording_model(self, recording):
        return build_spatio_temporal_recording_model(recording)

    def build_annotation_model(self):
        annotation = super().build_annotation_model()
        return build_timed_annotation_model(annotation)
