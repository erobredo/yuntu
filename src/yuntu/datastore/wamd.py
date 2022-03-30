import chunk
import struct

from guano import GuanoFile, tzoffset
import pytz
from datetime import datetime

from yuntu.datastore.base import Storage
from yuntu.core.audio.utils import hash_file, read_info, media_open
from yuntu.core.database.recordings import ULTRASONIC_SAMPLERATE_THRESHOLD

# WAMD field identifiers
WAMD_IDS = {
    0x00: 'version',
    0x01: 'model',
    0x02: 'serial',
    0x03: 'firmware',
    0x04: 'prefix',
    0x05: 'timestamp',
    0x06: 'gpsfirst',
    0x07: 'gpstrack',
    0x08: 'software',
    0x09: 'license',
    0x0A: 'notes',
    0x0B: 'auto_id',
    0x0C: 'manual_id',
    0x0D: 'voicenotes',
    0x0E: 'auto_id_stats',
    0x0F: 'time_expansion',
    0x10: 'program',
    0x11: 'runstate',
    0x12: 'microphone',
    0x13: 'sensitivity',
}

# Fields that should be excluded from our in-memory representation
WAMD_DROP_IDS = (
    0x0D,    # voice note embedded .WAV
    0x10,    # program binary
    0x11,    # runstate giant binary blob
    0xFFFF,  # used for 16-bit alignment
)

# rules to coerce values from binary string to native types (default is `str`)
WAMD_COERCE = {
    'version': lambda x: struct.unpack('<H', x)[0],
    'timestamp': lambda x: _parse_wamd_timestamp(x),
    'gpsfirst': lambda x: _parse_wamd_gps(x),
    'time_expansion': lambda x: struct.unpack('<H', x)[0],
}

def _parse_text(value):
    """Default coercion function which assumes text is UTF-8 encoded"""
    return value.decode('utf-8')

def _parse_wamd_timestamp(timestamp):
    """WAMD timestamps are one of these known formats:
    2014-04-02 22:59:14-05:00
    2014-04-02 22:59:14.000
    2014-04-02 22:59:14
    Produces a `datetime.datetime`.
    """
    if isinstance(timestamp, bytes):
        timestamp = timestamp.decode('utf-8')
    if len(timestamp) == 25:
        dt, offset = timestamp[:-6], timestamp[19:]
        tz = tzoffset(offset)
        return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz)
    elif len(timestamp) == 23:
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    elif len(timestamp) == 19:
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    else:
        return None


def _parse_wamd_gps(gpsfirst):
    """WAMD "GPS First" waypoints are in one of these two formats:
    SM3, SM4, (the correct format):
        WGS..., LAT, N|S, LON, E|W [, alt...]
    EMTouch:
        WGS..., [-]LAT, [-]LON[,alt...]
    Produces (lat, lon, altitude) float tuple.
    """
    if not gpsfirst:
        return None
    if isinstance(gpsfirst, bytes):
        gpsfirst = gpsfirst.decode('utf-8')
    vals = tuple(val.strip() for val in gpsfirst.split(','))
    datum, vals = vals[0], vals[1:]
    if vals[1] in ('N', 'S'):
        # Standard format
        lat, lon = float(vals[0]), float(vals[2])
        if vals[1] == 'S':
            lat *= -1
        if vals[3] == 'W':
            lon *= -1
        alt = int(round(float(vals[4]))) if len(vals) > 4 else None
    else:
        # EMTouch format
        lat, lon = float(vals[0]), float(vals[1])
        alt = int(round(float(vals[2]))) if len(vals) > 2 else None
    return lat, lon, alt

def read_wamd_info(path):
    """Extract WAMD metadata from a .WAV file as a dict"""
    with media_open(path, 'rb') as f:
        ch = chunk.Chunk(f, bigendian=False)
        if ch.getname() != b'RIFF':
            raise Exception('%s is not a RIFF file!' % path)
        if ch.read(4) != b'WAVE':
            raise Exception('%s is not a WAVE file!' % path)

        wamd_chunk = None
        while True:
            try:
                subch = chunk.Chunk(ch, bigendian=False)
            except EOFError:
                break
            if subch.getname() == b'wamd':
                wamd_chunk = subch
                break
            else:
                subch.skip()
        if not wamd_chunk:
            raise Exception('"wamd" WAV chunk not found in file %s' % path)

        metadata = {}
        offset = 0
        size = wamd_chunk.getsize()
        buf = wamd_chunk.read(size)

        while offset < size:
            id = struct.unpack_from('< H', buf, offset)[0]
            len = struct.unpack_from('< I', buf, offset+2)[0]
            val = struct.unpack_from('< %ds' % len, buf, offset+6)[0]
            if id not in WAMD_DROP_IDS:
                name = WAMD_IDS.get(id, id)
                val = WAMD_COERCE.get(name, _parse_text)(val)
                metadata[name] = val
            offset += 6 + len
        return metadata


class WAMDStorage(Storage):

    def iter_annotations(self, datum):
        return []

    def get_metadata(self):
        meta = {"type": "WAMDStorage"}
        meta["dir_path"] = self.dir_path

        return meta

    def prepare_datum(self, datum):
        wamd_info = read_wamd_info(datum)
        media_info = read_info(datum, timeexp=1.0)
        samplerate = media_info['samplerate']

        spectrum = 'ultrasonic' if samplerate > ULTRASONIC_SAMPLERATE_THRESHOLD else 'audible'

        timezone_utc = pytz.timezone("UTC")
        datetime_ = wamd_info.pop('timestamp')

        time_zone = datetime_.tzinfo.tzname(datetime_)
        time_format = '%H:%M:%S %d/%m/%Y (%Z)'
        time_raw = datetime_.strftime(format=time_format)

        return {
            'path': datum,
            'hash': hash_file(datum),
            'timeexp': 1,
            'media_info': media_info,
            'metadata': wamd_info,
            'spectrum': spectrum,
            'time_raw': time_raw,
            'time_format': time_format,
            'time_zone': time_zone,
            'time_utc': datetime_.astimezone(timezone_utc)
        }

    def prepare_annotation(self, datum, annotation):
        pass
