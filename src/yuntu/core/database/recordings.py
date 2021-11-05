"""Model fabric for recordings."""
from pony.orm import Required
from pony.orm import Optional
from pony.orm import PrimaryKey
from pony.orm import Set
from pony.orm import Json
from datetime import datetime


AUDIBLE_SPECTRUM = 'audible'
ULTRASONIC_SPECTRUM = 'ultrasonic'
ULTRASONIC_SAMPLERATE_THRESHOLD = 100000
SPECTRUMS = [AUDIBLE_SPECTRUM, ULTRASONIC_SPECTRUM]


def build_base_recording_model(db):
    """Create base recording model."""
    class Recording(db.Entity):
        """Basic recording entity for yuntu."""
        id = PrimaryKey(int, auto=True)
        datastore = Optional('Datastore')
        path = Required(str, unique=True)
        hash = Required(str, unique=True)
        timeexp = Required(float)
        spectrum = Required(str)
        media_info = Required(Json)
        annotations = Set('Annotation')
        metadata = Required(Json)

        def before_insert(self):
            if self.spectrum not in SPECTRUMS:
                message = (
                    f"Invalid spectrum. Options are: {AUDIBLE_SPECTRUM}"
                    f", {ULTRASONIC_SPECTRUM}.")
                raise ValueError(message)

            samplerate = self.media_info["samplerate"] * self.timeexp
            if self.spectrum == AUDIBLE_SPECTRUM:
                if samplerate > ULTRASONIC_SAMPLERATE_THRESHOLD:
                    raise ValueError("Not an audible recording.")

            if self.spectrum == ULTRASONIC_SPECTRUM:
                if samplerate <= ULTRASONIC_SAMPLERATE_THRESHOLD:
                    print(samplerate)
                    raise ValueError("Not an ultrasonic recording.")

    return Recording


def build_timed_recording_model(Recording):
    class TimedRecording(Recording):
        """Datastore that builds data from a foreign database."""
        time_raw = Required(str)
        time_format = Required(str)
        time_zone = Required(str)
        time_utc = Required(datetime, precision=6)

    return TimedRecording
