import os
import json
import shutil
import datetime,time
from yuntu.core.audio.utils import binaryMD5
from yuntu.core.db.base import embeddedDb
from yuntu.core.db.methods import lDbUpdateField
from yuntu.core.datastore.base import simpleDatastore, directDatastore,mongoDatastore
from yuntu.collection.utils import audioIterator, audioArray, cleanDirectory

def collectionPersistParser(col,parserDict):
    path = parserDict["path"]
    fname = binaryMD5(path)+".py"
    newPath = os.path.join(col.colPath,"parsers",fname)
    shutil.copy(path,newPath)
    creation =  time.strftime("%d-%m-%Y,%H:%M:%S", time.gmtime())

    parserDict["original_path"] = path
    parserDict["path"] = newPath
    parserDict["creation"] = creation

    return parserDict

def collectionPath(col):
    name = col.name
    dirPath = col.dirPath
    return os.path.join(dirPath,name)

def collectionExists(col):
    colPath = col.colPath
    if os.path.exists(colPath):
        return True
    else:
        return False

def collectionDrop(col):
    colPath = col.colPath
    parserPath = os.path.join(colPath,"parsers")
    mediaPath = os.path.join(colPath,"media")
    dbPath = os.path.join(colPath,"db")

    if cleanDirectory(colPath) and cleanDirectory(parserPath) and cleanDirectory(mediaPath) and cleanDirectory(dbPath):
        return True
    else:
        raise ValueError("Cannot clean all directories in path. More permissions may be needed")

def collectionLoadInfoFile(name,dirPath):
    infoPath = os.path.join(dirPath,name,"info.json")
    with open(infoPath) as infile:
        info = json.load(infile)
    return info

def collectionLoadInfo(col):
    col.info = collectionLoadInfoFile(col.name,col.dirPath)
    
    return True

def collectionSaveInfo(col):
    with open(os.path.join(col.colPath,"info.json"), 'w') as outfile:
        json.dump(col.info, outfile)

    return True

def collectionBuild(col):
    colPath = col.colPath
    dbPath = os.path.join(colPath,"db")
    parserPath = os.path.join(colPath,"parsers")
    if not os.path.exists(colPath):
        os.mkdir(colPath)
    if not os.path.exists(dbPath):
        os.mkdir(dbPath)
    if not os.path.exists(parserPath):
        os.mkdir(parserPath)
    col.db = embeddedDb(col.name,os.path.join(col.colPath,"db"),True)

    strtime = time.strftime("%d-%m-%Y,%H:%M:%S", time.gmtime())
    col.info["connPath"] = col.db.connPath
    col.info["creation"] = strtime
    #col.info["modification"] = strtime

    return collectionSaveInfo(col)

def collectionLoad(col):
    collectionLoadInfo(col)
    col.db = embeddedDb(col.name,os.path.join(col.colPath,"db"),False)
    return True

def collectionInsert(col,input,parseSeq):
    for i in range(len(parseSeq)):
        parseSeq[i] = collectionPersistParser(col,parseSeq[i])

    if isinstance(input,simpleDatastore):
        ds = input
    else:
        ds = directDatastore(input)

    return col.db.insert(ds.getData(),parseSeq)

def collectionPullDatastore(col,dsDict,parseSeq):
    if dsDict["type"] == "mongodb":
        ds = mongoDatastore(dsDict)
        return col.insertMedia(ds,parseSeq)
    else:
        raise ValueError("Datastore not implemented")

def collectionTransform(col,parseSeq,id=None,where=None,query=None,operation="append"):
    for i in range(len(parseSeq)):
        parseSeq[i] = colMethods.collectionPersistParser(col,parseSeq[i])

    return col.db.transform(parseSeq,id,where,query,operation)

def collectionDropMedia(col,id=None,where=None,query=None):
    removed = col.db.remove(id,where,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if os.path.isdir(internal_data_dir):
        for row in removed:
            if row["path"] != row["original_path"]:
                os.remove(row["path"])
    return removed

def collectionQuery(col,id=None,where=None,query=None,iterate=True):
    if where is not None:
        matches = col.db.select(id,where)
    else:
        matches = col.db.find(id,query)

    if iterate:
        return audioIterator(matches)
    else:
        return audioArray(matches)

def collectionDump(col,dirPath,overwrite=False):
    oldColPath = col.colPath
    newColPath = os.path.join(dirPath,col.name)

    if oldColPath == newColPath:
        return oldColPath

    if os.path.exists(newColPath):
        if overwrite:
            shutil.rmtree(newColPath)
            try:
                shutil.copytree(oldColPath, newColPath)
            except OSError as e:
                if e.errno == errno.ENOTDIR:
                    shutil.copy(oldColPath, newColPath)
                else:
                    print('Collection not dumped. Error: %s' % e)
        else:
            raise ValueError("Collection directory exists. Please set overwrite=True.")

    return newColPath

def collectionMaterialize(col,dirPath=None,overwrite=False):
    col.db.close()

    if dirPath is not None:
        newColPath = collectionDump(col,dirPath,overwrite)
    else:
        newColPath = col.colPath

    newMediaPath = os.path.join(newColPath,"media")

    if not os.path.exists(newMediaPath):
        os.mkdir(newMediaPath)

    newDb = embeddedDb(col.name,os.path.join(newColPath,"db"),False)
    matches = newDb.select()

    for row in matches:
        md5 = row["md5"]
        orid = row["orid"]
        oldPath = row["path"]

        newPath = os.path.join(newMediaPath,md5+".wav")



        if newPath != oldPath:
            shutil.copyfile(oldPath,newPath)
            lDbUpdateField(newDb,"original_path",oldPath,orid)
            lDbUpdateField(newDb,"path",newPath,orid)
            newMediaInfo = dict(row["media_info"])
            newMediaInfo["path"] = newPath
            lDbUpdateField(newDb,"media_info",json.dumps(newMediaInfo),orid)

    newDb.close()
    col.db.connect()

    return newColPath




















