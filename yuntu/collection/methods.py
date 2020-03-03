import os
import csv
import json
import shutil
import datetime,time
from  tempfile import NamedTemporaryFile
from yuntu.core.db.base import embeddedDb,RAMDb
from yuntu.core.db.methods import lDbUpdateField,lDbTimedInsert
from yuntu.core.db.utils import addTimeFields,loadParser
from yuntu.core.datastore.base import simpleDatastore, directDatastore,mongodbDatastore,audioMothDatastore,postgresqlDatastore
from yuntu.collection.server import yuntuServer
from yuntu.collection.utils import audioIterator, audioArray, metadataIterator, metadataArray, specIterator, specArray, buildColDirStruct
from yuntu.core.common.utils import loadJsonFile,dumpJsonFile,binaryMD5,cleanDirectory


def collectionPersistParser(col,parserDict):
    path = parserDict["path"]
    fname = binaryMD5(path)+".py"
    creation =  time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    parserDict["original_path"] = path
    parserDict["creation"] = creation

    if not col.virtual:
        newPath = os.path.join(col.colPath,"parsers",fname)
        shutil.copy(path,newPath)
        parserDict["path"] = fname
    else:
        func = loadParser(parserDict)
        pFile = open(parserDict["path"])
        source = [line for line in pFile]
        pFile.close()
        parserDict["path"] = fname
        col.db.parsers[fname] = {"parserDict":parserDict,"source":source,"func":func}


    return parserDict


def collectionPath(col):
    name = col.name
    dirPath = col.dirPath
    return os.path.abspath(os.path.join(dirPath,name))

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

def collectionLoadInfo(col):
    infoPath = os.path.join(col.dirPath,col.name,"info.json")
    col.info = loadJsonFile(infoPath)

    return True

def collectionSaveInfo(col):
    infoPath = os.path.join(col.colPath,"info.json")
    return dumpJsonFile(infoPath,col.info)

def collectionBuildDirStructure(colPath,parts=["db","parsers","sql"]):
    dbPath = os.path.join(colPath,"db")
    parserPath = os.path.join(colPath,"parsers")
    sqlPath = os.path.join(colPath,"sql")

    if not os.path.exists(colPath):
        os.mkdir(colPath)
    if not os.path.exists(dbPath):
        os.mkdir(dbPath)
    if not os.path.exists(parserPath):
        os.mkdir(parserPath)
    if not os.path.exists(sqlPath):
        os.mkdir(sqlPath)

    return True

def collectionBuild(col,dbDump=None):
    if buildColDirStruct(col.colPath):
        col.db = embeddedDb(col.name,os.path.join(col.colPath,"db"),dbDump,True)
        col.db.parsersDir = os.path.join(col.colPath,"parsers")
        strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
        col.info["connPath"] = os.path.abspath(col.db.connPath)
        col.info["creation"] = strtime
        col.info["dirPath"] = os.path.abspath(col.info["dirPath"])

        return collectionSaveInfo(col)

def collectionLoad(col):
    collectionLoadInfo(col)
    dbDump=None
    if os.path.splitext(col.info["connPath"])[1] == ".sql":
        dbDump = col.info["connPath"]
    col.db = embeddedDb(col.name,os.path.join(col.colPath,"db"),dbDump,False)
    col.db.parsersDir = os.path.join(col.colPath,"parsers")
    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    col.info["connPath"] = os.path.abspath(col.db.connPath)
    col.info["creation"] = strtime
    col.info["dirPath"] = os.path.abspath(col.info["dirPath"])

    return True

def collectionVirtualInit(col,dbDump=None):
    col.db = RAMDb(col.name,dump=dbDump)
    col.db.parsers = {}
    col.db.parsersDir = None
    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    col.info["connPath"] = ":memory:"
    col.info["creation"] = strtime

    return True

def collectionAnnotate(col,dataArr):
    return col.db.annotate(dataArr)

def collectionInsert(col,input,parseSeq):
    for i in range(len(parseSeq)):
        parseSeq[i] = collectionPersistParser(col,parseSeq[i])

    if isinstance(input,simpleDatastore):
        ds = input
    else:
        ds = directDatastore(input)

    return col.db.insert(ds.getData(),parseSeq)

def collectionTimedInsert(col,input,parseSeq):
    for i in range(len(parseSeq)):
        parseSeq[i] = collectionPersistParser(col,parseSeq[i])

    if isinstance(input,simpleDatastore):
        ds = input
    else:
        ds = directDatastore(input)
    return lDbTimedInsert(col.db,ds.getData(),parseSeq,col.timeField,col.tzField,col.timeFormat)

def collectionPullDatastore(col,dsDict,parseSeq):
    if dsDict["type"] == "mongodb":
        ds = mongodbDatastore(dsDict)
        return col.insertMedia(ds,parseSeq)
    elif dsDict["type"] == "audioMoth":
        ds = audioMothDatastore(dsDict)
        return col.insertMedia(ds,parseSeq)
    elif dsDict["type"] == "postgresql":
        ds = postgresqlDatastore(dsDict)
        return col.insertMedia(ds,parseSeq)
    else:
        raise ValueError("Datastore not implemented")

def collectionTransform(col,parseSeq,id=None,where=None,query=None,operation="append"):
    for i in range(len(parseSeq)):
        parseSeq[i] = collectionPersistParser(col,parseSeq[i])

    return col.db.transform(parseSeq,id,where,query,operation)

def collectionDropMedia(col,id=None,where=None,query=None):
    removed = col.db.remove(id,where,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if os.path.isdir(internal_data_dir):
        for row in removed:
            path = row["path"]
            if os.path.dirname(path) != "":
                if path != row["original_path"]:
                    os.remove(path)
            else:
                os.remove(os.path.join(internal_data_dir,path))
    return removed

def collectionQuery(col,id=None,query=None,iterate=True):
    matches = col.db.find(id,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if iterate:
        return audioIterator(matches,internal_data_dir)
    else:
        return audioArray(matches,internal_data_dir)

def collectionGetAnnotations(col,noteid=None,query=None,iterate=True):
    if noteid is not None:
        query = {"id":noteid}
    matches = col.db.find(query=query,table="annotations")
    
    if iterate:
        return annotationIterator(matches)
    else:
        return matches

def collectionGetMetadata(col,id=None,query=None,iterate=True):
    matches = col.db.find(id,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if iterate:
        return metadataIterator(matches,internal_data_dir)
    else:
        return metadataArray(matches,internal_data_dir)

def collectionGetSignals(col,id=None,query=None,readSr=None,iterate=True):
    matches = col.db.find(id,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if iterate:
        return signalIterator(matches,internal_data_dir,readSr)
    else:
        return signalArray(matches,internal_data_dir,readSr)

def collectionGetSpecs(col,id=None,query=None,readSr=None,n_fft=1024,hop_length=512,iterate=True):
    matches = col.db.find(id,query)
    internal_data_dir = os.path.join(col.colPath,"media")
    if iterate:
        return specIterator(matches,internal_data_dir,readSr,n_fft,hop_length)
    else:
        return specArray(matches,internal_data_dir,readSr,n_fft,hop_length)

def collectionBuildResampledMedia(col,dirPath,out_sr=24000,media_format="wav",chop=60,thresh=1,orid=None,query=None,iterate=False):
    rsdir = "sr_"+str(out_sr)
    if col.virtual and dirPath is None:
        raise ValueError("'dirPath' is mandatory for virtual collections.")
    elif not col.virtual and dirPath is None:
        out_dir = os.path.join(col.colPath,"media",rsdir)
    else:
        out_dir = os.path.join(dirPath,rsdir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    media = col.getMedia(orid=orid,query=query,iterate=iterate)

    if iterate:
        def f():
            for au in media:
                yield au.writeChunks(basePath=os.path.join(out_dir,au.md5),chop=chop,thresh=thresh,media_format=media_format,sr=out_sr)
        return f()
    else:
        return [au.writeChunks(basePath=os.path.join(out_dir,au.md5),chop=chop,thresh=thresh,media_format=media_format,sr=out_sr) for au in media]


def collectionDump(col,dirPath,overwrite=False):
    if col.virtual:
        return collectionVirtualDump(col,dirPath,overwrite)

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

def collectionDoggyBag(col,dirPath="",outName=None,overwrite=True):
    if outName is None:
        outName = "doggy_bag_"+col.name
    doggy_path = os.path.join(dirPath,outName)

    if os.path.exists(doggy_path):
        if not overwrite:
            raise ValueError("Output path exists but overwrite is False")
    else:
        os.makedirs(doggy_path)

    doggy_media_dir = os.path.join(doggy_path,"media")
    if not os.path.exists(doggy_media_dir):
        os.makedirs(doggy_media_dir)

    matches = col.db.select()
    fieldnames = list(set(["path","md5"]+["fname"]+[key for key in matches[0]["metadata"].keys()]))
    with open(os.path.join(doggy_path,'metadata.csv'), mode='w') as f:
        writer = csv.DictWriter(f, delimiter='|', quotechar='"', fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in matches:
            oriPath = row["path"]

            if os.path.dirname(oriPath) == "":
                fname = oriPath
                oriPath = os.path.join(col.colPath,"media",oriPath)
            else:
                fname = os.path.basename(oriPath)

            md5 = row["md5"]
            metadata = row["metadata"]
            metadata["md5"] = md5
            metadata["path"] = os.path.join("media",fname)
            metadata["fname"] = fname

            for key in metadata.keys():
                if isinstance(metadata[key],dict):
                    metadata[key] = json.dumps(metadata[key])

            writer.writerow(metadata)
            shutil.copyfile(oriPath,os.path.join(doggy_media_dir,fname))



def collectionMaterialize(col,dirPath=None,overwrite=False):
    if col.virtual:
        if dirPath is None:
            raise ValueError("'dirPath' must be defined for a virtual collection materialization.")
        return collectionVirtualMaterialize(col,dirPath,overwrite)

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

        if os.path.dirname(path) == "":
            oldPath = os.path.join(col.colPath,"media",oldPath)

        newName = md5+".wav"
        newPath = os.path.join(newMediaPath,newName)

        if newPath != oldPath:
            shutil.copyfile(oldPath,newPath)
            lDbUpdateField(newDb,"original_path",oldPath,orid)
            lDbUpdateField(newDb,"path",newName,orid)
            newMediaInfo = dict(row["media_info"])
            newMediaInfo["path"] = newName
            lDbUpdateField(newDb,"media_info",json.dumps(newMediaInfo),orid)

    newDb.close()
    col.db.connect()

    return newColPath

def collectionParsersToDisk(col,parserDir):
    parsers = col.db.parsers
    pKeys = list(parsers.keys())
    for p in pKeys:
        source = parsers[p]["source"]
        outPath = os.path.join(parserDir,p)
        out = open(outPath,"w")
        for line in source:
            out.write(line)
        out.close()


def collectionPersistVirtualStructure(col,dirPath,overwrite):
    newColPath = os.path.join(dirPath,col.name)
    if buildColDirStruct(newColPath):
        dinfo = col.info.copy()
        strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
        collectionParsersToDisk(col,os.path.join(newColPath,"parsers"))
        sqlPath = col.db.dump(os.path.join(newColPath,"sql",col.name+"_"+strtime+".sql"),overwrite=overwrite)
        dinfo["connPath"] = os.path.abspath(sqlPath)
        dinfo["dirPath"] = dirPath
        return dumpJsonFile(os.path.join(newColPath,"info.json"),dinfo)

def collectionVirtualDump(col,dirPath,overwrite=False):
    if os.path.exists(os.path.join(dirPath,col.name)):
        if overwrite:
            return collectionPersistVirtualStructure(col,dirPath,overwrite)
        else:
            raise ValueError("Collection directory exists. Please set overwrite=True.")
    else:
        return collectionPersistVirtualStructure(col,dirPath,overwrite)

def collectionVirtualMaterialize(col,dirPath,overwrite=False):
    newColPath = os.path.join(dirPath,col.name)
    if buildColDirStruct(newColPath):
        newMediaPath = os.path.join(newColPath,"media")

        if not os.path.exists(newMediaPath):
            os.mkdir(newMediaPath)

        for row in col.db.select():
            md5 = row["md5"]
            orid = row["orid"]
            oldPath = row["path"]
            newPath = os.path.join(newMediaPath,md5+".wav")

            if newPath != oldPath:
                shutil.copyfile(oldPath,newPath)
                lDbUpdateField(col.db,"original_path",oldPath,orid)
                lDbUpdateField(col.db,"path",newPath,orid)
                newMediaInfo = dict(row["media_info"])
                newMediaInfo["path"] = newPath
                lDbUpdateField(col.db,"media_info",json.dumps(newMediaInfo),orid)

        return collectionVirtualDump(col,dirPath,overwrite)

def collectionToDisk(col,dirPath,overwrite=False):
    if col.virtual:
        newColPath = os.path.join(dirPath,col.name)
        if collectionVirtualDump(col,dirPath,overwrite):
            statements = col.db.asStatements()
            col.db.close()
            col.__init__(col.name,dirPath=dirPath,metadata=col.metadata,virtual=False,dbDump=statements,overwrite=overwrite)
            return True
    else:
        raise ValueError("Collection is not virtual")

def collectionGetSize(col,where=None,query=None):
    return col.db.count(where,query)

def collectionBuildTime(col):
    for row in addTimeFields(col.db.find(),col.timeField,col.tzField,col.timeFormat):
        lDbUpdateField(col.db,"metadata",json.dumps(row["metadata"]),orid=row["orid"])

    col.info["type"] = "timedCollection"
    col.info["timeField"] = col.timeField
    col.info["tzField"] = col.tzField

    return collectionSaveInfo(col)


def collectionServer(col,port=9797):
    return yuntuServer(col,port)
