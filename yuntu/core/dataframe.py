import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from audio import Audio
from utils import binaryMD5
from tqdm import tqdm

def indexAudio(audioArr,metadataParse=None):
    data = {}
    total_audio = len(audioArr)
    print("Indexing audio...")

    pbar = tqdm(total=total_audio)
    for i in range(total_audio):
        audio = audioArr[i]

        if metadataParse is not None:
            metadata = metadataParse(audio.metadata)
        else:
            metadata = audio.metadata

        if "md5" in metadata:
            md5 = metadata["md5"]
        elif "akey" in metadata:
            md5 = metadata["akey"]
            metadata["md5"] = md5
        else:
            md5 = binaryMD5(audio.path)
            metadata["md5"] = md5

        metadata["akey"] = md5
        audio.set_metadata(metadata)
        data[md5] = audio

        pbar.update(1)

    audio_ids = data.keys()
    dictArr = [data[key].metadata for key in audio_ids]

    if len(audio_ids) < total_audio:
        print("Droped "+str(total_audio-len(audio_ids))+" duplicates (md5 match).")

    return data,dictArr

def fromArray(dictArr):
    return pd.DataFrame.from_dict(dictArr)

def fromAudioArray(audioArr,metadataParse=None):
    data, dictArr = indexAudio(audioArr,metadataParse)
    return data, fromArray(dictArr)

def fromMongoQuery(mongoDict,metadataParse):
    pathField = "path"
    timeexpField = None
    
    if "pathField" in mongoDict:
        pathField = mongoDict["pathField"]
    if "timeexpField" in mongoDict:
        timeexpField = mongoDict["timeexpField"]
        
    client = MongoClient(mongoDict["host"],maxPoolSize = 30)
    mongoDb = client[mongoDict["db"]]
    collection = mongoDb[mongoDict["collection"]]

    audioArr = []
    for metadata in collection.find(mongoDict["query"]).limit(10):
        if "media_info" in metadata:
            config = metadata["media_info"]
        else:
            path = metadata[pathField]
            timeexp = 1.0
            if timeexpField is not None:
                timeexp = metadata[timeexpField]

            config = {"path":path,"timeexp":timeexp}

        audioArr.append(Audio(config,metadata=metadata))

    if len(audioArr) == 0:
        raise ValueError("No matches in mongo query.")

    return fromAudioArray(audioArr,metadataParse)

def dropData(df,condition=None):
    if condition is None:
        indArr = list(df["akey"])
    else:
        indArr = list(df[condition]["akey"])
    df = df[~df["akey"].isin(indArr)]
    return df, indArr

def upsertData(df,audioArr,metadataParse=None):
    dictArr, data = indexAudio(audioArr,metadataParse)
    return data, df.append(dictArr)

def vectorGroupBy(df,target,agg_func,groupBy,condition=None):
    if condition is None:
        result = df.groupby(groupBy).agg({target:lambda x:list(agg_func(x))})
    else:
        result = df[condition].groupby(groupBy).agg({target:lambda x:list(agg_func(x))})

    result[target] = result[target].apply(np.array)

    return result

def vectorApply(df,target,new_col_name,map_func,condition=None):
    if condition is None:
        result = df[target].map(lambda x: map_func(x))
    else:
        result = df[condition][target].map(lambda x: map_func(x))

    df[new_col_name] = result
    return df

def makeSeries(obj,index):
    return pd.Series(obj,index)