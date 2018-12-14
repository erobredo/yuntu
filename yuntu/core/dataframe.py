import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
from audio import Audio
from utils import binaryMD5

def indexAudio(audioArr,metadataParse=None):
    data = {}
    dictArr = []
    for i in range(audioArr):
        audio = audioArr[i]

        if metadataParse is not None:
            metadata = metadataParse(audio.metadata)
        else:
            metadata = audio.metadata

        md5 = binaryMD5(audio.path)
        metadata["akey"] = md5
        audio.set_metadata(metadata)
        data[md5] = audio
        dictArr.append(metadata)

    return data,dictArr

def fromArray(dictArr):
    return pd.DataFrame.from_dict(dictArr)

def fromAudioArray(audioArr,metadataParse=None):
    data, dictArr = indexAudio(audioArr,metadataParse)
    return data, fromArray(dictArr)

def fromMongoQuery(mongoDict,metadataParse,pathField="path",timeexpField=None):
    client=MongoClient(mongoDict["host"],maxPoolSize = 30)
    mongoDb = client[mongoDict["db"]]
    collection = mongoDb[mongoDict["collection"]]

    audioArr = []
    for metadata in collection.find(query):
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

    return fromAudioArray(audioArr)

def dropData(df,condition):
    indArr = list(df[condition]["akey"])
    df = df[~df["akey"].isin(indArr)]
    return df, indArr

def upsertData(df,audioArr,metadataParse=None):
    dictArr, data = indexAudio(audioArr,metadataParse)
    return data, df.append(dictArr)

def vectorGroupBy(df,target,agg_func,groupBy,condition=True):
    result = df[condition].groupby(groupBy).agg({target:lambda x:list(agg_func(x))})
    result[target] = result[target].apply(np.array)
    return

def vectorApply(df,target,new_col_name,map_func,condition=True):
    result = df[target].map(lambda x: map_func(x))
    df[new_col_name] = result
    return df