from pymongo import MongoClient
from bson.objectid import ObjectId
from yuntu.core.datastore.utils import hashDict

def datastoreGetSpec(ds):
    dSpec = {}
    dSpec["hash"] = ds.getHash()
    dSpec["type"] = ds.getType()
    dSpec["conf"] = ds.getConf()
    dSpec["metadata"] = ds.getMetadata()

    return dSpec

def datastoreGetType(ds):
    return ds.inputSpec["type"]

def datastoreGetConf(ds):
    dConf = {}
    for key in ["host","datastore","target","filter","fields","ukey"]:
        dConf[key] = ds.inputSpec["conf"][key]

    return dConf

def datastoreGetMetadata(ds):
    return ds.inputSpec["metadata"]

def datastoreGetHash(ds):
    formatedConf = ds.getConf()
    return hashDict(formatedConf)

def datastoreMongoGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        client = MongoClient(dsConf["host"],maxPoolSize = 30)
        mDb = client[dsConf["datastore"]]
        collection = mDb[dsConf["target"]]

        for obj in collection.find(dsConf["filter"],dsConf["fields"]):
            fkey = str(obj[dsConf["ukey"]])
            obj[dsConf["ukey"]] = fkey
            yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}

    return f(ds.getSpec())

def datastoreDirectGetData(ds):
    def f(dsSpec,dataArr):
        for i in range(len(dataArr)):
            yield  {"datastore":dsSpec,"source":{"fkey":i},"metadata":dataArr[i]}

    return f(ds.getSpec(),ds.dataArr)


