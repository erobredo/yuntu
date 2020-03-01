import os
import datetime,time
import pytz
from pytz import timezone
from yuntu.core.common.utils import loadMethodFromFile
from yuntu.core.audio.base import Audio
from yuntu.core.common.utils import binaryMD5


def loadParser(parserDict,parsersDir=None):
    path = parserDict["path"]
    if parsersDir is not None:
        if os.path.dirname(path) != "":
            raise ValueError("'parsersDir' provided but 'path' already has a dirname")
        path = os.path.join(parsersDir,path)

    return loadMethodFromFile(path,parserDict["function"])

def sequentialTransform(meta,parseSeq,parsers,parsersDir=None):
    #print(parsers)
    metadata = meta.copy()
    if parsers is None:
        for parserDict in parseSeq:
            parserFunction = loadParser(parserDict,parsersDir)
            metadata = parserFunction(metadata)
            if "kwargs" in parserDict:
                metadata = parserFunction(metadata,**parserDict["kwargs"])
            else:
                metadata = parserFunction(metadata)
    else:
        for parserDict in parseSeq:
            pkey = parserDict["path"]
            if pkey in parsers:
                parserFunction = parsers[pkey]["func"]
                if "kwargs" in parserDict:
                    metadata = parserFunction(metadata,**parserDict["kwargs"])
                else:
                    metadata = parserFunction(metadata)
            else:
                raise ValueError("Parser not in 'parsers' dict.")

    return metadata

def describeAudio(path,timeexp,md5):
    au = Audio({"path":path,"timeexp":timeexp})
    if md5 is None:
        md5 = binaryMD5(path)

    return au.getMediaInfo(),md5

def standardizeTime(strTime,timeZone,format='%d-%m-%Y %H:%M:%S'):
    tzObj = pytz.timezone(timeZone)
    
    return tzObj.localize(datetime.datetime.strptime(strTime,format))

def transformTime(locTime,newTz):
    return locTime.astimezone(tz=timezone(newTz))
    

def timeAsSeconds(dTime):
    tzObj = pytz.timezone("UTC")
    globalStart = tzObj.localize(datetime.datetime(1970,1,1,0,0,0))
    
    return (dTime-globalStart).total_seconds()

def stripName(string):
    new_string = ""
    for character in string:
        if character == " ":
            new_string += "_"
        elif character.isalnum():
            new_string += character
            
    return new_string

def catToStrTime(cat,unit,startDate,timeZone="UTC",timeFormat='%d-%m-%Y %H:%M:%S',transformTz=None,outFormat="%H:%M"):
    seconds = cat*unit
    standardStart = standardizeTime(startDate,timeZone,timeFormat)

    if transformTz:
        dtime = transformTime(standardStart,transformTz)  + datetime.timedelta(seconds=seconds)
        return dtime.strftime(outFormat)
    else:
        return standardStart.strftime(outFormat)


def timeAsCat(standardStartStr,startDate,timeZone,timeFormat,unit,modulo,transformTz=None):
    modulo_seconds = unit*modulo
    standardStart = datetime.datetime.fromisoformat(standardStartStr)
    
    if transformTz is not None:
        standardAbsStart = transformTime(standardizeTime(startDate,timeZone,timeFormat),transformTz)
    else:
        standardAbsStart = standardizeTime(startDate,timeZone,timeFormat)
            
    delta_from_start = standardStart-standardAbsStart
    remainder = delta_from_start % datetime.timedelta(seconds=modulo_seconds)
    
    return int(round(remainder/datetime.timedelta(seconds=unit))) % modulo
    

# def timeAsCat(absStart,absStop,startDate,timeZone,timeFormat,unit,modulo,transformTz=None):
    
#     if transformTz is not None:
#         standardAbsStart = timeAsSeconds(transformTime(standardizeTime(startDate,timeZone,timeFormat),transformTz))
#     else:
#         standardAbsStart = timeAsSeconds(standardizeTime(startDate,timeZone,timeFormat))
        
#     return int(round((float(absStart+absStop)/2 - standardAbsStart)/unit))%modulo

def getTimeFields(metadata,media_info,timeField,tzField,format):

    strTime = metadata[timeField]
    timeZone = metadata[tzField]
    duration = media_info["duration"]
    
    standardStart = standardizeTime(strTime,timeZone,format)
    absStart = timeAsSeconds(standardStart)

    standardStop = standardStart + datetime.timedelta(seconds=duration)
    absStop = timeAsSeconds(standardStop)

    metadata["standardStart"] = standardStart.isoformat(' ')
    metadata["standardStop"] = standardStop.isoformat(' ')
    metadata["absStart"] = absStart
    metadata["absStop"] = absStop

    return metadata

def addTimeFields(dataArr,timeField,tzField,format):
    def f(data):
        for row in data:
            media_info = row["media_info"]
            metadata = row["metadata"]
            yield getTimeFields(metadata,media_info,timeField,tzField,format)

    return f(dataArr)

def jsonExtractFormat(field,parentField):
    return "json_extract("+parentField+",'$."+field+"')"

