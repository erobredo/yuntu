from yuntu.core.audio.base import Audio
from yuntu.core.audio.utils import binaryMD5

import importlib.util

def loadParser(parserDict):
    #File must be inside "parser" directory
    spec = importlib.util.spec_from_file_location(parserDict["function"],parserDict["path"])
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)

    return parser.Parser

def describeAudio(path,timeexp,md5):
    au = Audio({"path":path,"timeexp":timeexp})
    if md5 is None:
        md5 = binaryMD5(path)

    return au.get_media_info(),md5


