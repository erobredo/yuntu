'''Timed database manager'''
from yuntu.core.database.annotations import build_timed_annotation_model
from yuntu.core.database.recordings import build_timed_recording_model
from yuntu.core.database.base import DatabaseManager

class TimedDatabaseManager(DatabaseManager):
    def build_recording_model(self):
        recording = super().build_recording_model()
        return build_timed_recording_model(recording)

    def build_annotation_model(self):
        annotation = super().build_annotation_model()
        return build_timed_annotation_model(annotation)
