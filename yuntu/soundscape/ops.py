import datetime
import numpy as np
import pandas as pd
import json
import dask.bag as daskBag
import dask.dataframe as dd
from dask.threaded import get
from dask.delayed import delayed
from dask.optimization import cull,inline,inline_functions,fuse
from yuntu.core.db.utils import timeAsCat
from yuntu.core.audio.base import Audio


def writeParquet(node,path):
    print("Writing parquet to "+path)
    return node.to_parquet(path,compression="GZIP")


def readParquet(path,npartitions):
    return dd.read_parquet(path)

def projectMetadata(obj,fields):
    newObj = {}
    newObj["orid"] = obj["orid"]
    newObj["media_info"] = obj["media_info"]
    newObj["fileDuration"] = obj["media_info"]["duration"]

    for key in fields:
        key_split = key.split(".")
        if len(key_split) > 1:
            
            val = obj["metadata"][key_split[0]]
            for i in range(1,len(key_split)):
                val = val[key_split[i]]
                
            newObj[key] = val
        else:
            if key in obj["metadata"]:
                newObj[key] = obj["metadata"][key]

    return newObj

def doTransform(e,eTransform,transformDict):
    if eTransform is not None:
        kwargs = {}
        if "kwargs" in transformDict:
            if transformDict["kwargs"] is not None:
                kwargs = transformDict["kwargs"]

        return eTransform(e,**kwargs)
    else:
        return e
    
    
def sliceEnergy(spec,eCols,fCuts,duration,start,stop,unitSize,eTransform,config):
    results = {}
    try:
        fbins,tbins = spec.shape
        if stop - start > config["tStep"]:
            e = spec
            nbins = tbins
        else:
            startBin = min(int(round((float(tbins)/duration)*start)),tbins-1)
            stopBin = min(startBin+unitSize,tbins-1)
            nbins = stopBin-startBin
            e = spec[:,startBin:stopBin]

        if nbins < float(unitSize)/2:
            raise ValueError("Generated empty slice!")

        results["chunkTbins"] = nbins
        results["chunkWeight"] = float(nbins)/unitSize

        if eTransform["full_spec"] != "pass":
            tConfig = config["transformations"]["full_spec"].copy()
            if "kwargs" in tConfig:
                #Use available context variables
                for kname in tConfig["kwargs"]:
                    if tConfig["kwargs"][kname] == "$fCuts":
                        tConfig["kwargs"][kname] = fCuts

            global_results =  doTransform(e,eTransform["full_spec"],tConfig)
            #Ensure transformation returns a dictionary with correct keys.
            for i in range(len(eCols)):
                results[eCols[i]] = global_results[eCols[i]]

            return results

        if eTransform["spec"] != "pass":
            e = doTransform(e,eTransform["spec"],config["transformations"]["spec"])
        #e = np.sum(e,axis=1)

        for i in range(len(eCols)):
            cut = fCuts[i]
            if cut[0] != cut[1]:
                eCut = e[cut[0]:cut[1],:]
                eCut = doTransform(eCut,eTransform["aggr"],config["transformations"]["aggr"])
            else:
                eCut = None

            results[eCols[i]] = eCut
    except:
        for col in ["chunkTbins","chunkWeight"]+eCols:
            results[col] = 0

    return results

def splitTime(obj,eCols,eTransform,config):
    prevCols = [key for key in list(obj.keys()) if key != "media_info"]
    tStep = config["tStep"]
    media_info = obj["media_info"]
    fileDuration = media_info["duration"]
    sTransform = eTransform["signal"]

    au = Audio(config=media_info,fromConfig=True)
    au.unsetMask()

    sr = media_info["samplerate"]
    if config["readSr"] is not None:
        if config["readSr"] != media_info["samplerate"]:
            sr = config["readSr"]
            au.setReadSr(config["readSr"])

    unitSize = int(round(float(sr*tStep-config["n_fft"])/config["hop_length"])+1)
    nsteps = int(round(fileDuration/tStep))

    chanList = [0]
    
    if config["channels"] is None:
        chanList = [None]
    elif config["channels"] == "__all__":
        if media_info["nchannels"] > 1:
            chanList = range(media_info["nchannels"])
    elif isinstance(config["channels"],list):
        chanList = config["channels"]
        if len(chanList) > media_info["nchannels"]:
            raise ValueError("Too many channels")
    else:
        raise ValueError("'channel' param should be list,str or None")
            
    preProcess = None
    if sTransform != "pass":
        kwargs = {}
        if "kwargs" in config["transformations"]["signal"]:
            if config["transformations"]["signal"]["kwargs"] is not None:
                kwargs = config["transformations"]["signal"]["kwargs"]

        preProcess = {"transform":sTransform,"kwargs":kwargs}

    out = []
    for chan in chanList:
        #try:
        spec, freqs = au.getSpec(chan,config["n_fft"],config["hop_length"],preProcess=preProcess)
        
        if chan is None:
            chan = 99
        
        topFreq = np.amax(freqs)
        fbins = freqs.size
        maxFreqIdx = fbins
        minFreqIdx = 0
        freqRange = topFreq

        if config["fLimits"] is not None:
            freqRange = config["fLimits"][1]-config["fLimits"][0]
            maxFreqIdx = min(int(round(fbins*config["fLimits"][1]/freqRange)),fbins)
            minFreqIdx = min(int(round(fbins*config["fLimits"][0]/freqRange)),fbins)

            if maxFreqIdx <= minFreqIdx:
                raise ValueError("Wrong frequency limits")

            spec = spec[minFreqIdx:maxFreqIdx,:]
        fStep = round(int(float(freqRange)/config["fBins"]))
        fSteps = [[i*fStep,(i+1)*fStep] for i in range(config["fBins"])]
        fCuts =  [[(np.abs(freqs - x[0])).argmin(),(np.abs(freqs - x[1])).argmin()] for x in fSteps]



        #except:
        #    print("Error during energy processing...")
        #    spec = None

        if spec is not None:
            for step in range(nsteps):
                raw = {}
                for key in prevCols:
                    raw[key] = obj[key]

                fileStart = step*tStep
                fileStop = min(fileStart+tStep,obj["fileDuration"])
                duration = fileStop-fileStart

                startSec = fileStart+obj["absStart"]
                stopSec = startSec+duration

                raw["channel"] = chan
                raw["chunkId"] = str(obj["orid"])+"-"+str(startSec)+"-"+str(stopSec)
                raw["chunkFileStart"] = fileStart
                raw["chunkFileStop"] = fileStop
                raw["chunkDuration"] = duration
                raw["chunkAbsStart"] = startSec
                raw["chunkAbsStop"] = stopSec
                
                eResults = sliceEnergy(spec,eCols,fCuts,fileDuration,fileStart,fileStop,unitSize,eTransform,config)
                for col in ["chunkTbins","chunkWeight"]+eCols:
                    raw[col] = eResults[col]
                out.append(raw)

    au.clearMedia()

    return out


def calendarize(row,config,fieldName,transformTz):
    row[fieldName] = timeAsCat(row["standardStart"],config["startDate"],config["timeZone"],config["timeFormat"],config["unit"],config["modulo"],transformTz)
    
    return row

def loadFragment(data,groupfields,config):
    columns = ["orid","media_info","fileDuration","standardStart","standardStop","absStart","absStop"]+groupfields

    return daskBag.from_sequence(data,npartitions=config["npartitions"]).map(projectMetadata,columns)

def makeSplits(fragment,basicmeta,splitmeta,groupmeta,eCols,eTransform,config):
    meta = basicmeta+splitmeta+groupmeta
    meta.append(("channel",np.dtype('int64')))

    for col in eCols:
        meta.append((col, np.dtype('float64')))

    dobjs = [delayed(splitTime)(row,eCols,eTransform,config) for row in fragment]

    return daskBag.from_delayed(dobjs).to_dataframe(meta=meta)


def makeCalendar(splits,config):
    meta = [x for x in zip(splits.columns.values,splits.dtypes.values)]+[("timeCell",np.dtype('int64'))]

    return splits.apply(calendarize,meta=meta,axis=1,config=config,fieldName="timeCell",transformTz=None)

def makeCronoCounts(splits):
    return splits.groupby(['timeCell'])["orid"].agg(['count'])

def makeCounts(splits):
    return len(splits)

def makeSample(splits,groupfields,config):
    if config["size"] == -1:
        return splits
    else:
        groupcols = ["timeCell"]
        if groupfields is not None:
            groupcols += groupfields

        meta = [x for x in zip(splits.columns.values,splits.dtypes.values)]

        return splits.groupby(groupcols).apply(lambda x: x.sample(frac=float(config["size"])/len(x),replace=config["replace"]) if len(x) > config["size"] else x,meta=meta).reset_index(drop=True)

def makeStats(statKey):
    def f(sample):
        return sample

    return f

def dGet(graph,opList,client):
    if client is not None:
        return client.get(graph,opList,sync=False)
    else:
        return get(graph,opList)

def linearizeOps(graph,opList):
    graph1, deps = cull(graph, opList)
    graph2 = inline(graph1, dependencies=deps)
    graph3 = inline_functions(graph2, opList, [len, str.split],dependencies=deps)
    graph4, deps = fuse(graph3)

    return graph4


