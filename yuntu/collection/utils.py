import os
from yuntu.core.audio.base import Audio

def audioIterator(dataArr,mediaDir):
    for row in dataArr:
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            row["media_info"]["path"] = os.path.join(mediaDir,path)

        yield Audio(row["media_info"],fromConfig=True)

def audioArray(dataArr,mediaDir):
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)

    return [Audio(row["media_info"],fromConfig=True) for row in dataArr]

def metadataArray(dataArr,mediaDir):
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)

    return dataArr

def metadataIterator(dataArr,mediaDir):
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)

        yield dataArr[i]

def annotationIterator(dataArr):
    for i in range(len(dataArr)):
        yield dataArr[i]

def signalArray(dataArr,mediaDir,readSr):
    results = []
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)
        au = Audio(row["media_info"],fromConfig=True)
        if readSr is not None:
            au.setReadSr(readSr)
        signal = au.getSignal()
        au.clearMedia()
        results.append({"id":dataArr[i]["orid"],"md5":dataArr[i]["md5"],"signal":signal})
        
    return results

def signalIterator(dataArr,mediaDir,readSr):
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)
        au = Audio(row["media_info"],fromConfig=True)
        if readSr is not None:
            au.setReadSr(readSr)
        signal = au.getSignal()
        au.clearMedia()

        yield {"id":dataArr[i]["orid"],"md5":dataArr[i]["md5"],"signal":signal}

def specArray(dataArr,mediaDir,readSr,n_fft,hop_length):
    results = []
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)
        au = Audio(row["media_info"],fromConfig=True)
        if readSr is not None:
            au.setReadSr(readSr)
        freqs,spec = au.getSpec(n_fft=n_fft,hop_length=hop_length)
        au.clearMedia()
        results.append({"id":dataArr[i]["orid"],"md5":dataArr[i]["md5"],"freqs":freqs,"spec":spec})
        
    return results

def specIterator(dataArr,mediaDir,readSr,n_fft,hop_length):
    for i in range(len(dataArr)):
        row = dataArr[i]
        path = row["media_info"]["path"]
        if os.path.dirname(path) == "":
            dataArr[i]["media_info"]["path"] = os.path.join(mediaDir,path)
        au = Audio(row["media_info"],fromConfig=True)
        if readSr is not None:
            au.setReadSr(readSr)
        freqs,spec = au.getSpec(n_fft=n_fft,hop_length=hop_length)
        au.clearMedia()
        
        yield {"id":dataArr[i]["orid"],"md5":dataArr[i]["md5"],"freqs":freqs,"spec":spec}

def buildColDirStruct(colPath,parts=["db","parsers","sql"]):
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

