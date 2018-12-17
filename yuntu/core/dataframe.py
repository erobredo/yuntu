import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from audio import Audio
from utils import binaryMD5
from tqdm import tqdm
import os
import shutil
import json
import datetime,time

def is_number(s):
    try:
        complex(s) 
    except ValueError:
        return False

    return True

def csv_to_dict(filename,delimiter='\t'):
    
    dict_arr = []
    with open(filename) as f:
        reader = csv.reader(f,delimiter=delimiter)
        
        header = next(reader)
        
        for row in reader:
            row_dict = {}
            for i in range(len(row)):
                val = row[i]
                if is_number(val):
                    val = float(val)

                row_dict[header[i]] = val
            dict_arr.append(row_dict)

    return dict_arr

def timeToSeconds(timeDict):
    dtime = datetime.datetime(timeDict["year"], timeDict["month"], timeDict["day"], timeDict["hour"], timeDict["minute"], timeDict["second"])
    return time.mktime(dtime.timetuple())

def validateTimeFields(meta):
    timeFields = ["year","month","day","hour","minute","second"]
    if len(set(timeFields)&set(meta.keys())) >= len(timeFields):
        absTime = timeToSeconds(meta)
        meta["absTime"] = absTime
        return meta
    else:
        raise ValueError("Validation error: time fields missing.")

def indexAudio(audioArr,metadataParse=None,validateTime=False):
    print("Indexing audio...")

    data = {}
    total_audio = len(audioArr)
    

    pbar = tqdm(total=total_audio)
    for i in range(total_audio):
        audio = audioArr[i]

        if metadataParse is not None:
            metadata = metadataParse(audio.metadata)
        else:
            metadata = audio.metadata

        if validateTime:
            metadata = validateTimeFields(metadata)
                

        if "md5" in metadata:
            md5 = metadata["md5"]
        else:
            md5 = binaryMD5(audio.path)
            metadata["md5"] = md5


        audio.set_metadata(metadata)

        metadata["path"] = audio.path
        metadata["timeexp"] = audio.timeexp

        data[md5] = audio

        pbar.update(1)

    audio_ids = data.keys()
    dictArr = [data[key].metadata for key in audio_ids]

    if len(audio_ids) < total_audio:
        print("Droped "+str(total_audio-len(audio_ids))+" duplicates (md5).")

    return data, dictArr

def fromArray(dictArr):
    return pd.DataFrame.from_dict(dictArr)

def fromAudioArray(audioArr,metadataParse=None,validateTime=False):
    data, dictArr = indexAudio(audioArr,metadataParse,validateTime)
    return data, fromArray(dictArr)


def dropData(df,condition=None):
    if condition is None:
        indArr = list(df["md5"])
    else:
        indArr = list(df[condition]["md5"])
    df = df[~df["md5"].isin(indArr)]
    return df, indArr

def upsertData(df,audioArr,metadataParse=None,validateTime=False):
    dictArr, data = indexAudio(audioArr,metadataParse,validateTime)
    return data, df.append(dictArr)

def vectorApply(df,target,new_col_name,map_func,condition=None):
    if condition is None:
        result = df[target].map(lambda x: map_func(x))
    else:
        result = df[condition][target].map(lambda x: map_func(x))

    df[new_col_name] = result
    return df

def makeSeries(obj,index):
    return pd.Series(obj,index)

def makeBaseDir(basePath,overwrite=False):
    dataDir = os.path.join(basePath,"data")
    configDirPath = os.path.join(basePath,'config')
    metadataDirPath = os.path.join(basePath,'metadata')

    if os.path.exists(dataDir):
        if not overwrite:
            raise ValueError("File exists. Use 'overwrite' parameter.")
        else:
            shutil.rmtree(dataDir)
            os.makedirs(dataDir)
            return basePath

    else:
        os.makedirs(dataDir)
        return basePath

def loadConfig(basePath):
    configPath = os.path.join(basePath,'config','collection_config.json')
    with open(configPath) as infile:
        config = json.load(infile)
    return config

def loadMetadata(basePath):
    metadataPath = os.path.join(basePath,'metadata','collection_metadata.json')
    with open(metadataPath) as infile:
        metadata = json.load(infile)
    return metadata

def loadDataFrame(basePath):
    config = loadConfig(basePath)
    metadataAll = loadMetadata(basePath)

    metadata = []
    data = {}
    for meta in metadataAll:
        media_info = meta["media_info"]
        data[meta["metadata"]["md5"]] = Audio(media_info,metadata=meta["metadata"])
        metadata.append(meta["metadata"])

    df = fromArray(metadata)
    size = len(metadata)

    return data, df, size, config

def dumpMetadata(data,df,basePath,config,overwrite=False):
    configDirPath = os.path.join(basePath,'config')
    metadataDirPath = os.path.join(basePath,'metadata')

    configPath = os.path.join(configDirPath,'collection_config.json')
    metadataPath = os.path.join(metadataDirPath,'collection_metadata.json')

    if os.path.exists(configPath):
        if not overwrite:
            raise ValueError("Configuration path contains information. Set overwrite=True (this will not delete files in an existing 'data' directory, only configuration and metadata).")
        else:
            shutil.rmtree(configDirPath)
            os.makedirs(configDirPath)
    else:
        os.makedirs(configDirPath)

    if os.path.exists(metadataPath):
        if not overwrite:
            raise ValueError("Metadata path contains information. Set overwrite=True (this will not delete files in an existing 'data' directory, only configuration and metadata).")
        else:
            shutil.rmtree(metadataDirPath)
            os.makedirs(metadataDirPath)
    else:
        os.makedirs(metadataDirPath)


    metadataRaw = df.to_dict('records')
    metadata = []
    for i in range(len(metadataRaw)):
        md5 = metadataRaw[i]["md5"]
        audio = data[md5]
        metadata.append({"metadata":metadataRaw[i],"media_info":audio.get_media_info()})

    with open(configPath, 'w') as outfile:
        json.dump(config, outfile)
    with open(metadataPath, 'w') as outfile:
        json.dump(metadata, outfile)

    return basePath

def dumpDataFrame(data,df,basePath,config,media_format="wav",overwrite=False):
    dataPath = os.path.join(basePath,'data')
    configDirPath = os.path.join(basePath,'config')
    metadataDirPath = os.path.join(basePath,'metadata')

    configPath = os.path.join(configDirPath,'collection_config.json')
    metadataPath = os.path.join(metadataDirPath,'collection_metadata.json')


    if os.path.exists(configPath):
        if not overwrite:
            raise ValueError("Configuration path contains information. Set overwrite=True (this WILL DELETE ALL files in the collection directory).")
        else:
            shutil.rmtree(configDirPath)
            os.makedirs(configDirPath)
    else:
        os.makedirs(configDirPath)

    if os.path.exists(metadataPath):
        if not overwrite:
            raise ValueError("Metadata path contains information. Set overwrite=True (this WILL DELETE ALL files in the collection directory).")
        else:
            shutil.rmtree(metadataDirPath)
            os.makedirs(metadataDirPath)
    else:
        os.makedirs(metadataDirPath)

    if os.path.exists(dataPath):
        if not overwrite:
            raise ValueError("Data path contains information. Set overwrite=True (this WILL DELETE ALL files in the collection directory).")
        else:
            shutil.rmtree(dataPath)
            os.makedirs(dataPath)
    else:
        os.makedirs(dataPath)

    metadataRaw = df.to_dict('records')
    metadata = []
    for i in range(len(metadataRaw)):
        md5 = metadataRaw[i]["md5"]
        audio = data[md5]
        writePath = os.path.join(dataPath,md5+"."+media_format)
        shutil.copyfile(metadataRaw[i]["path"],writePath)
        metadataRaw[i]["original_path"] = metadataRaw[i]["path"]
        metadataRaw[i]["path"] = writePath
        media_info = audio.get_media_info()
        media_info["path"] = metadataRaw[i]["path"]
        metadata.append({"metadata":metadataRaw[i],"media_info":media_info})
        

    with open(configPath, 'w') as outfile:
        json.dump(config, outfile)
    with open(metadataPath, 'w') as outfile:
        json.dump(metadata, outfile)


    return basePath

def arr2audio(arr,pathField="path",timeexpField=None):
    audioArr = []
    for meta in arr:
        path = meta[pathField]
        timeexp = 1.0
        if timeexpField is not None:
            timeexp = meta[timeexpField]

        config = {"path":path,"timeexp":timeexp}

        audioArr.append(Audio(config,metadata=meta))

    if len(audioArr) == 0:
        raise ValueError("No data in array.")

    return audioArr


def mongo2audio(dbConfig,pathField="path",timeexpField=None):        
    client = MongoClient(dbConfig["host"],maxPoolSize = 30)
    mongoDb = client[dbConfig["db"]]
    collection = mongoDb[dbConfig["collection"]]

    audioArr = []
    for meta in collection.find(dbConfig["query"]):
        if "media_info" in meta:
            config = meta["media_info"]
        else:
            path = meta[pathField]
            timeexp = 1.0
            if timeexpField is not None:
                timeexp = meta[timeexpField]

            config = {"path":path,"timeexp":timeexp}

        audioArr.append(Audio(config,metadata=meta))

    if len(audioArr) == 0:
        raise ValueError("No matches in mongo query.")

    return audioArr


def csv2audio(filePath,delimiter=",",pathField="path",timeexpField=None):
    arr = csv_to_dict(filePath,delimiter=delimiter)
    return arr2audio(arr,pathField,timeexpField)
    

#def sqlite2audio(dbConfig,pathField="path",timeexpField=None):
#    pass

#def pg2audio(dbConfig):
#    pass


