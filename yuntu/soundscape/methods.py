import os
import time,datetime
import numpy as np
import yuntu.soundscape.ops as scOps
from yuntu.core.common.utils import cleanDirectory,dumpJsonFile,loadJsonFile
from yuntu.soundscape.utils import loadTransform, getCombinations, filterExpr
from yuntu.core.db.utils import jsonExtractFormat
from yuntu.core.db.methods import lDbSelect,lDbParseQuery
from yuntu.collection.base import timedCollection,simpleCollection


def soundscapeBuild(sc):
    baseDir = os.path.join(sc.dirPath,sc.name)
    persistDir = os.path.join(baseDir,"persist")

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    if not os.path.exists(persistDir):
        os.makedirs(persistDir)

    cleanDirectory(baseDir,dirs=False)
    cleanDirectory(persistDir)

    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    sc.info["creation"] = strtime
    sc.info["collection"] = sc.collection.info
    sc.info["type"] = sc.getType()

    sc.config = sc.getDefaultConfig()


    return True


def soundscapeLoad(sc):
    sc.config = sc.getDefaultConfig()
    sc.info = loadJsonFile(os.path.join(sc.dirPath,sc.name,"info.json"))
    sc.info["type"] = sc.getType()
    colInfo = sc.info["collection"]

    if sc.collection is not None:
        if colInfo["dirPath"] == sc.collection.info["dirPath"] and sc.collection.info["name"] == colInfo["name"]:
            return True
        else:
            raise ValueError("Explicit collection does not match collection defined previously. Did you mean to overwrite soundscape?")
    else:
        return soundscapeLoadCollection(sc,colInfo)
        


def soundscapeLoadCollection(sc,colInfo):
    dirPath = colInfo["dirPath"]
    name = colInfo["name"]
    if colInfo["type"] == "timedCollection":
        tzField = colInfo["tzField"]
        timeField = colInfo["timeField"]
        sc.collection = timedCollection(name=name,dirPath=dirPath,timeField=timeField,tzField=tzField)
    else:
        sc.collection = simpleCollection(name=name,dirPath=dirPath)

    return True



def soundscapeExists(sc):
    return os.path.exists(os.path.join(sc.dirPath,sc.name,"info.json"))

def soundscapeSetConfig(sc,config):
    if config is not None:
        if "globalParams" in config:
            soundscapeSetGlobalParams(sc,config["globalParams"])
        if "energyParams" in config:
            soundscapeSetEnergyParams(sc,config["energyParams"])
        if "samplingParams" in config:
            soundscapeSetSamplingParams(sc,config["samplingParams"])
        if "indexParams" in config:
            soundscapeSetIndexparams(sc,config["indexParams"])
        if "cronoParams" in config:
            soundscapeSetCronoParams(sc,config["cronoParams"])

    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    sc.info["modification"] = strtime
    sc.info["config"] = sc.config
    sc.info["dirPath"] = os.path.abspath(sc.info["dirPath"])

    dumpJsonFile(os.path.join(sc.dirPath,sc.name,"info.json"),sc.info)

    return sc.loadGraph()

def soundscapeSetIndexparams(sc,indexParams):
    for pname in sc.config["indexParams"]:
        if pname in indexParams:
            sc.config["indexParams"][pname] = indexParams[pname] 


def soundscapeGetFieldLevels(sc,field):
    selectSt = "DISTINCT "+jsonExtractFormat(field,"metadata")+" AS levels FROM parsed "
    if sc.config["globalParams"]["collectionFilter"] is not None:
        selectSt += "WHERE "+lDbParseQuery(sc.config["globalParams"]["collectionFilter"])
    distinct =  lDbSelect(sc.collection.db,freeSt=selectSt)
    vals = []
    for row in distinct:
        vals.append(row["levels"])

    if len(vals) == 0:
        raise ValueError("No levels for field "+field)

    return vals


def soundscapeGetGroupCount(sc,group):
    groupingFields = sc.config["globalParams"]["groupingFields"]
    val = group[0]
    if isinstance(val,str):
        val = "'"+val+"'"
    selectSt = "count(*) as gcount FROM parsed WHERE "+jsonExtractFormat(groupingFields[0],"metadata")+"="+str(val)
    for i in range(1,len(groupingFields)):
        val = group[i]
        if isinstance(val,str):
            val = "'"+val+"'"
        selectSt += " AND "+jsonExtractFormat(groupingFields[i],"metadata")+"="+str(val)
    dcount = lDbSelect(sc.collection.db,freeSt=selectSt)

    return dcount[0]["gcount"]

def soundscapeInferMetadataTypes(sc,groupingFields):
    selectSt = "metadata from parsed"
    if sc.config["globalParams"]["collectionFilter"] is not None:
        selectSt += " WHERE "+lDbParseQuery(sc.config["globalParams"]["collectionFilter"])

    selectSt += " limit 10"
    metaSample = lDbSelect(sc.collection.db,freeSt=selectSt)

    dTypes = {}
    for field in groupingFields:
        dTypes[field] = []
    
    for row in metaSample:
        metadata = row["metadata"]
        for field in dTypes:
            infType = type(metadata[field]).__name__
            if infType not in dTypes[field]:
                dTypes[field].append(infType)

    typeArr = []
    for i in range(len(groupingFields)):
        if len(dTypes[groupingFields[i]]) > 1:
            raise ValueError("Type not consistent for field "+field+". Got types : "+str(dTypes))
        typeArr.append(dTypes[groupingFields[i]][0])


    return typeArr

def soundscapeConfigureLevels(sc):
    groupingFields = sc.config["globalParams"]["groupingFields"]
    groupingTypes = sc.config["globalParams"]["groupingTypes"]
    doGroup = False
    if groupingFields is not None:
        if len(groupingFields) > 0:
            if groupingTypes is None:
                sc.config["globalParams"]["groupingTypes"] = soundscapeInferMetadataTypes(sc,groupingFields)
            doGroup = True

    levels = {}
    if doGroup:
        for field in groupingFields:
            levels[field] = soundscapeGetFieldLevels(sc,field)

        groups = getCombinations([levels[field] for field in groupingFields])

        for group in groups:
            gCount = soundscapeGetGroupCount(sc,group)
            if gCount == 0:
                print("No entries for group "+str(group))

    sc.levels = levels

    return True


def soundscapeSetGlobalParams(sc,globalParams):
    for pname in sc.config["globalParams"]:
        if pname in globalParams:
            sc.config["globalParams"][pname] = globalParams[pname]


    print("Ensure collection fragment is not empty...")
    colSize = sc.collection.getSize(query=sc.config["globalParams"]["collectionFilter"])

    if colSize["count"] > 0:
        print("Fragment size: "+str(colSize["count"]))
    else:
        raise ValueError("No items in fragment.")

    return soundscapeConfigureLevels(sc)


def soundscapeSetEnergyParams(sc,energyParams):
    for pname in sc.getDefaultConfig()["energyParams"]:
        if pname in energyParams:
            sc.config["energyParams"][pname] = energyParams[pname]
    return True

def soundscapeSetAcousticIndicesParams(sc,indexParams):
    for pname in sc.getDefaultConfig()["indexParams"]:
        if pname in indexParams:
            sc.config["indexParams"][pname] = indexParams[pname]
    return True

def soundscapeSetSamplingParams(sc,samplingParams):
    for pname in sc.getDefaultConfig()["samplingParams"]:
        if pname in samplingParams:
            sc.config["samplingParams"][pname] = samplingParams[pname]
    return True

def soundscapeSetCronoParams(sc,cronoParams):
    for pname in sc.getDefaultConfig()["cronoParams"]:
        if pname in cronoParams:
            sc.config["cronoParams"][pname] = cronoParams[pname]
    return True

def soundscapeSetNode(sc,name,opDef,isOutput=False):
    sc.graph[name] = opDef
    if isOutput:

        sc.outs.append(name)

    return True

def soundscapeGetGroupSpecs(sc):
    gFields = sc.config["globalParams"]["groupingFields"]
    gMeta = sc.config["globalParams"]["groupingTypes"]

    dMeta = []
    dFields = []
    if gFields is not None:
        for i in range(len(gFields)):
            if gMeta[i] == "str":
                dMeta.append((gFields[i],np.dtype('O')))
            else:
                dMeta.append((gFields[i],np.dtype(gMeta[i])))
            dFields.append(gFields[i])

    return dMeta,dFields

def soundscapeSetLoadNodes(sc):
    sc.setNode("basicMeta",[('orid', np.dtype('int64')), ('fileDuration', np.dtype('float64')), ('standardStart', np.dtype('O')), ('standardStop', np.dtype('O')), ('absStart', np.dtype('float64')), ('absStop', np.dtype('float64'))])
    sc.setNode("splitMeta",[('chunkId', np.dtype('O')),('chunkFileStart', np.dtype('float64')), ('chunkFileStop', np.dtype('float64')), ('chunkDuration', np.dtype('float64')),('chunkTbins',np.dtype('int64')),('chunkWeight', np.dtype('float64'))])
    dMeta, dFields = soundscapeGetGroupSpecs(sc)
    sc.setNode("groupMeta",dMeta)
    sc.setNode("groupFields",dFields)

    if sc.getType() == "cronoSoundscape":
        sc.graph["splitMeta"] += [('chunkAbsStart', np.dtype('float64')), ('chunkAbsStop', np.dtype('float64'))]
        sc.setNode("cronoParams",sc.config["cronoParams"])

    sc.setNode("globalParams",sc.config["globalParams"])
    sc.setNode("energyParams",sc.config["energyParams"])
    sc.setNode("samplingParams",sc.config["samplingParams"])
    sc.setNode("indexParams",sc.config["indexParams"])

    sigTransform = "pass"
    specTransform = "pass"
    aggrTransform = "pass"
    fullSpecTransform = "pass"

    if "signal" in sc.config["energyParams"]["transformations"]:
        sigTransform = loadTransform(sc.config["energyParams"]["transformations"]["signal"])
    if "spec" in sc.config["energyParams"]["transformations"]:
        specTransform = loadTransform(sc.config["energyParams"]["transformations"]["spec"])
    if "aggr" in sc.config["energyParams"]["transformations"]:
        aggrTransform = loadTransform(sc.config["energyParams"]["transformations"]["aggr"])
    if "full_spec" in sc.config["energyParams"]["transformations"]:
        fullSpecTransform = loadTransform(sc.config["energyParams"]["transformations"]["full_spec"])

    if not (fullSpecTransform != "pass" or aggrTransform != "pass"):
        raise ValueError("If a full spectrogram transformation is not specified, at least an aggregation transformation must be applied.")

    if fullSpecTransform != "pass":
        eCols = []
        if "out_names" not in sc.config["energyParams"]["transformations"]["full_spec"]:
            raise ValueError("'out_names' must be specified for full spectrum transformations")

        oNames = sc.config["energyParams"]["transformations"]["full_spec"]["out_names"]
        if "bin_based" in oNames:
            for base_name in oNames["bin_based"]:
                eCols += [base_name+"_"+str(i) for i in range(sc.config["energyParams"]["fBins"])]
        if "full" in oNames:
            for name in oNames["full"]:
                eCols.append(name)

        if len(eCols) == 0:
            raise ValueError("No names in sections 'bin_based' or 'full'. Please use the standard specification: {'bin_based':[<names>],'full':[<names>]}")
    else:
        eCols = ["e_"+str(i) for i in range(sc.config["energyParams"]["fBins"])]

    sc.setNode("eCols",eCols)
    sc.setNode("eTransform",{"signal":sigTransform,"spec":specTransform,"aggr":aggrTransform,"full_spec":fullSpecTransform})

    colFilter = sc.config["globalParams"]["collectionFilter"]
    dData = sc.collection.db.find(query=colFilter)

    sc.setNode("dataInput",dData)
    sc.setNode("fragment",(scOps.loadFragment,"dataInput","groupFields","globalParams"))
    sc.setNode("splits",(scOps.makeSplits,"fragment","basicMeta","splitMeta","groupMeta","eCols","eTransform","energyParams"),True)

    if sc.getType() == "cronoSoundscape":
        sc.setNode("calendarized",(scOps.makeCalendar,"splits","cronoParams"),True)

    return True

def soundscapeSetSamplingNodes(sc):
    if sc.getType() == "cronoSoundscape":
        sc.setNode("counts",(scOps.makeCronoCounts,"calendarized"),True)
        sc.setNode("sample", (scOps.makeSample,"calendarized","groupFields","samplingParams"),True)
    else:
        sc.setNode("counts",(scOps.makeCounts,"splits"),True)
        sc.setNode("sample",(scOps.makeSample,"splits","groupFields","samplingParams"),True)

    return True

def soundscapeSetStatNodes(sc):
    for statKey in ["mean","std","var"]:
        sc.setNode(statKey,(scOps.makeStats(statKey),"energy"),True)

    return True

def soundscapeWriteNode(sc,node,nodeName):
    path = os.path.join(sc.dirPath,sc.name,"persist",nodeName+".parquet")
    return scOps.writeParquet(node,path)

def soundscapeReadNode(sc,path):
    return scOps.readParquet(path,sc.config["globalParams"]["npartitions"])

def soundscapeAppendOp(sc,nodeName,op):
    pass

def soundscapeGetGroups(sc):
    return [name for name in list(sc.groupFilters.keys())]

def soundscapeGetConfig(sc):
    return sc.config

def soundscapeGetNode(sc,nodeName,group=None,compute=True,overwrite=False,):
    path = os.path.join(sc.dirPath,sc.name,"persist",nodeName+".parquet")
    nodeExists = os.path.exists(path)

    if nodeName in ["fragment","splits","calendarized"]:
         xgraph = scOps.linearizeOps(sc.graph,[nodeName])
    else:
         xgraph = sc.graph

    #xgraph = sc.graph

    if nodeName in sc.outs and nodeName != "sample":
        if compute:
            if overwrite or not nodeExists:
                node_ = scOps.dGet(xgraph,nodeName,sc.client)

                if sc.client is not None:
                    node_r = node_.result()
                    node = node_r.compute()
                else:
                    node = node_.compute()

                soundscapeWriteNode(sc,node,nodeName)
            else:
                node = soundscapeReadNode(sc,path).compute()
        else:
            print("Do not compute!")
            node = scOps.dGet(xgraph,nodeName,sc.client)

        if group is not None:
            for field in group:
                node = node[node[field]==group[field]]
            return node
        else:
            return node
    elif nodeName == "sample":
        node_ =  scOps.dGet(xgraph,nodeName,sc.client)
        if compute:
            if sc.client is not None:
                node_r = node_.result()
                node = node_r.compute()
            else:
                node = node_.compute()
        else:
            return node_
    else:
        return scOps.dGet(xgraph,nodeName,sc.client)


def soundscapeSetClient(sc,client):
    sc.client = client
    return True


# def soundscapeGetEnergy(sc,group=None,compute=True):
#     if sc.groupExists(group):
#         return soundscapeGetNode(sc,"energy",group,compute)

def soundscapeGetSplits(sc,group=None,compute=True):
    if sc.groupExists(group):
        op = "splits"
        if sc.getType() == "cronoSoundscape":
            op = "calendarized"
        return soundscapeGetNode(sc,op,group,compute)

def soundscapeGetSample(sc,group=None,compute=True,fromPersisted=True):
    if sc.groupExists(group):
        if fromPersisted:
            out = "splits.parquet"
            if sc.getType() == "cronoSoundscape":
                out = "calendarized.parquet"

            path = os.path.join(sc.dirPath,sc.name,"persist",out)

            if os.path.exists(path):
                splits = soundscapeReadNode(sc,path)
            else:
                print("No persisted splits available.")
                print("Building node from scratch...")

                return soundscapeGetNode(sc,"sample",group,compute)

            _, groupfields = soundscapeGetGroupSpecs(sc)
            config = sc.config["samplingParams"]
            sample = scOps.makeSample(splits,groupfields,config)

            if group is not None:
                for field in group:
                    sample = sample[sample[field]==group[field]]

            if compute:
                if sc.client is not None:
                    return sc.client.compute(sample).result()
                else:
                    return sample.compute()
            else:
                return sample

        else:
            return soundscapeGetNode(sc,"sample",group,compute)
    else:
        raise ValueError("Group does not exist")

def soundscapeGetCounts(sc,group=None,compute=True):
    if sc.groupExists(group):
        return soundscapeGetNode(sc,"counts",group,compute)

#def soundscapeGetStat(sc,stat):
#    return scOps.dGet(sc.graph,stat,sc.config["multiThread"])

def soundscapeGroupExists(sc,group):
    if group is None:
        return True
    for key in group:
        if key in sc.levels:
            if group[key] not in sc.levels[key]:
                ValueError("Value "+str(group[key])+" is not present within field "+key)
        else:
            raise ValueError("Field "+field+" was not declared as a grouping field")

    return True

def soundscapeGetAcousticIndices(sc):
    pass

def soundscapeDump(sc,name,overwrite):
    pass

def soundscapeSummary(sc):
    pass

def soundscapeClear(sc):
    #Dask needs more operations to clear memmory
    baseDir = os.path.join(sc.dirPath,sc.name)
    persistDir = os.path.join(baseDir,"persist")

    cleanDirectory(baseDir)
    cleanDirectory(persistDir)
    sc.graph = {}
    return True

def soundscapeSetCronoConfig(sc,config):
    sc.setCalendarizer(sc,config["cfonoParams"]["unit"],config["cfonoParams"]["modulo"],config["cfonoParams"]["start"])
    return soundscapeSetConfig(sc,config)


def soundscapeLoadGraph(sc):
    sc.graph = {}
    if soundscapeSetLoadNodes(sc):
        if soundscapeSetSamplingNodes(sc):
            return True

def soundscapeCompute(sc):
    for nodeName in sc.outs:
        soundscapeGetNode(nodeName,overwrite=True)







