import os
from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
import psycopg2
import psycopg2.extras
from collections import OrderedDict
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
    for key in ds.inputSpec["conf"]:
        dConf[key] = ds.inputSpec["conf"][key]

    return dConf

def datastoreGetMetadata(ds):
    return ds.inputSpec["metadata"]

def datastoreGetHash(ds):
    formatedConf = ds.getConf()

    return hashDict(formatedConf)

def datastorePostgresqlGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        conn = psycopg2.connect("dbname='"+dsConf["datastore"]+"' user='"+dsConf["user"]+"' host='"+dsConf["host"]+"' password='"+dsConf["password"]+"'")
        cur = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
        cur.execute(dsConf["target"])
        
        for row in cur:
            obj = {}
            fkey = str(row[dsConf["ukey"]])
            for key in row.keys():
                obj[key] = row[key]    
            obj[dsConf["ukey"]] = fkey
            
            yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}
            
    return f(ds.getSpec())

def datastoreMongodbGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        client = MongoClient(dsConf["host"],maxPoolSize = 30)
        mDb = client[dsConf["datastore"]]
        collection = mDb[dsConf["target"]]
 
        if isinstance(dsConf["filter"],list):
            for rId in dsConf["filter"]:
                obj = collection.find_one({"_id":ObjectId(rId)})
                fkey = str(obj[dsConf["ukey"]])
                obj[dsConf["ukey"]] = fkey
                for key in obj:
                    if isinstance(obj[key],ObjectId):
                        obj[key] = str(obj[key])
                    elif isinstance(obj[key],dict):
                        for dkey in obj[key]:
                            if isinstance(obj[key][dkey],ObjectId):
                                obj[key][dkey] = str(obj[key][dkey])
                            elif isinstance(obj[key][dkey],dict):
                                for tkey in obj[key][dkey]:
                                    if isinstance(obj[key][dkey][tkey],ObjectId):
                                        obj[key][dkey][tkey] = str(obj[key][dkey][tkey])
                                
                            
                yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}
        else:
            for obj in collection.find(dsConf["filter"],dsConf["fields"]):
                fkey = str(obj[dsConf["ukey"]])
                obj[dsConf["ukey"]] = fkey
                for key in obj:
                    if isinstance(obj[key],ObjectId):
                        obj[key] = str(obj[key])
                    elif isinstance(obj[key],dict):
                        for dkey in obj[key]:
                            if isinstance(obj[key][dkey],ObjectId):
                                obj[key][dkey] = str(obj[key][dkey])
                            elif isinstance(obj[key][dkey],dict):
                                for tkey in obj[key][dkey]:
                                    if isinstance(obj[key][dkey][tkey],ObjectId):
                                        obj[key][dkey][tkey] = str(obj[key][dkey][tkey])
                yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}

    return f(ds.getSpec())

def datastoreAudioMothGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        dataDir = dsConf["dataDir"]
        allFiles = []
        for filename in os.listdir(dsConf["dataDir"]):
            if filename.endswith(".wav") or filename.endswith(".WAV"): 
                allFiles.append(filename)

        for i in range(len(allFiles)):
            fkey = allFiles[i]
            obj = {}
            obj["path"] = os.path.join(dsConf["dataDir"],fkey)
            
            with open(obj["path"], 'rb') as file:
                buf_header = file.read(200)
                try:
                    obj["voltage"] = float(buf_header[166:169])
                    obj["time"] = buf_header[68:87].decode("utf-8")
                    obj["tZone"] = buf_header[89:92].decode("utf-8")
                    if "-" in buf_header[84:94].decode("utf-8"):
                        obj["tZone"] = buf_header[84:94].decode("utf-8")
                    obj["device_id"] = buf_header[107:123].decode("utf-8")
                    obj["gain"] = float(buf_header[140:141])
                except:
                    obj["voltage"] = float(buf_header[168:171])
                    obj["time"] = buf_header[68:87].decode("utf-8")
                    obj["tZone"] = buf_header[89:92].decode("utf-8")
                    if "-" in buf_header[84:94].decode("utf-8"):
                        obj["tZone"] = buf_header[84:94].decode("utf-8")
                    obj["device_id"] = buf_header[109:125].decode("utf-8")
                    obj["gain"] = float(buf_header[142:143])

                file.close()
                

            yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}

    return f(ds.getSpec())

def datastoreDirectGetData(ds):
    def f(dsSpec,dataArr):
        for i in range(len(dataArr)):
            yield  {"datastore":dsSpec,"source":{"fkey":i},"metadata":dataArr[i]}

    return f(ds.getSpec(),ds.dataArr)


