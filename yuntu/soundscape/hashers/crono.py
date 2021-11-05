"""Row hashers for category assingment."""
import numpy as np
import pandas as pd
from yuntu.soundscape.utils import aware_time, from_timestamp
import datetime
from yuntu.soundscape.hashers.base import Hasher

TIME_ZONE = "America/Mexico_city"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_START = "2018-01-01 00:00:00"
TIME_UNIT = 3600
TIME_MODULE = 24
TIME_COLUMN = "time_raw"
TIME_FORMAT_COLUMN = "time_format"
TIME_ZONE_COLUMN = "time_zone"
TIME_UTC_COLUMN = None
AWARE_START = aware_time(TIME_START, TIME_ZONE, TIME_FORMAT)

DEFAULT_HASHER_CONFIG = {
    "time_column": TIME_COLUMN,
    "tzone_column": TIME_ZONE_COLUMN,
    "format_column": TIME_FORMAT_COLUMN,
    "time_utc_column": TIME_UTC_COLUMN,
    "aware_start": AWARE_START,
    "start_time": TIME_START,
    "start_tzone": TIME_ZONE,
    "start_format": TIME_FORMAT,
    "time_unit": TIME_UNIT,
    "time_module": TIME_MODULE
}

class CronoHasher(Hasher):
    name = "crono_hasher"
    def __init__(self,
                 time_column=TIME_COLUMN,
                 tzone_column=TIME_ZONE_COLUMN,
                 format_column=TIME_FORMAT_COLUMN,
                 time_utc_column=TIME_UTC_COLUMN,
                 aware_start=AWARE_START,
                 start_time=TIME_START,
                 start_tzone=TIME_ZONE,
                 start_format=TIME_FORMAT,
                 time_unit=TIME_UNIT,
                 time_module=TIME_MODULE):

        if not isinstance(time_column, str):
            raise ValueError("Argument 'time_column' must be a string.")
        if not isinstance(tzone_column, str):
            raise ValueError("Argument 'tzone_column' must be a string.")
        if not isinstance(tzone_column, str):
            raise ValueError("Argument 'tzone_column' must be a string.")
        if not isinstance(format_column, str):
            raise ValueError("Argument 'format_column' must be a string.")

        if not isinstance(time_unit, (int, float)):
            raise ValueError("Argument 'time_unit' must be a number.")
        if not isinstance(time_module, (int, float)):
            raise ValueError("Argument 'time_module' must be a number.")

        if aware_start is not None:
            if not isinstance(aware_start, datetime.datetime):
                raise ValueError("Argument 'aware_start' must "
                                 "be a datetime.")
            self.start = aware_start

        else:
            if not isinstance(start_tzone, str):
                raise ValueError("Argument 'start_tzone' must be a string.")
            if not isinstance(start_time, str):
                raise ValueError("Argument 'start_time' must be a string.")
            if not isinstance(start_tzone, str):
                raise ValueError("Argument 'start_tzone' must be a string.")
            if not isinstance(start_format, str):
                raise ValueError("Argument 'start_format' must be a string.")
            self.start = aware_time(start_time, start_tzone, start_format)

        self.time_column = time_column
        self.tzone_column = tzone_column
        self.format_column = format_column
        self.time_utc_column = time_utc_column
        self.columns = [time_column, tzone_column, format_column]
        self.time_unit = time_unit
        self.time_module = time_module
        self.unit = datetime.timedelta(seconds=self.time_unit)
        self.module = datetime.timedelta(seconds=self.time_unit * self.time_module)

    def hash(self, row, out_name="crono_hash"):
        """Return rotating integer hash according to unit, module and start."""
        if self.time_utc_column is None:
            strtime = row[self.time_column]
            timezone = row[self.tzone_column]
            timeformat = row[self.format_column]
            atime = aware_time(strtime, timezone, timeformat)
        else:
            atime = from_timestamp(row[self.time_utc_column], tzone="UTC")

        delta_from_start = atime - self.start
        remainder = delta_from_start % self.module
        new_row = {}
        new_row[out_name] = np.int64(int(round(remainder/self.unit)) % self.time_module)

        return pd.Series(new_row)

    def unhash(self, hashed):
        """Produce datetime object according to input integer hash."""
        unit_seconds = self.unit.total_seconds()
        hash_delta = datetime.timedelta(seconds=hashed * unit_seconds)
        return self.start + hash_delta

    @property
    def dtype(self):
        return np.dtype('int64')
