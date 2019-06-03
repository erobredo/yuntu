
import os

def datasetExists(dset):
    return os.path.exists(os.path.join(dset.dirPath,dset.name,"info.json"))

def datasetBuild(dset):
    baseDir = os.path.join(dset.dirPath,dset.name)
    persistDir = os.path.join(baseDir,"persist")

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    if not os.path.exists(persistDir):
        os.makedirs(persistDir)

    cleanDirectory(baseDir,dirs=False)
    cleanDirectory(persistDir)

    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    dset.info["creation"] = strtime
    dset.info["collection"] = dset.collection.info
    dset.info["type"] = dset.getType()

    dset.config = dset.getDefaultConfig()


    return True


def datasetLoad(dset):
    dset.config = dset.getDefaultConfig()
    dset.info = loadJsonFile(os.path.join(dset.dirPath,dset.name,"info.json"))
    dset.info["type"] = dset.getType()
    dsetInfo = dset.info["collection"]

    if dset.collection is not None:
        if dsetInfo["dirPath"] == dset.collection.info["dirPath"] and dset.collection.info["name"] == dsetInfo["name"]:
            return True
        else:
            raise ValueError("Explicit collection does not match collection defined previously. Did you mean to overwrite dataset?")
    else:
        return dsetLoadCollection(dset,dsetInfo)

def datasetSetConfig(dset,config):
    if config is not None:
        if "globalParams" in config:
            datasetSetGlobalParams(sc,config["globalParams"])
        if "specParams" in config:
            datasetSetSpecParams(sc,config["energyParams"])
        if "splittingParams" in config:
            datasetSetSplittingParams(sc,config["samplingParams"])

    strtime = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    dset.info["modification"] = strtime
    dset.info["config"] = dset.config
    dset.info["dirPath"] = os.path.abspath(dset.info["dirPath"])

    dumpJsonFile(os.path.join(dset.dirPath,dset.name,"info.json"),dset.info)

    return dset.loadGraph()

def datasetSetGlobalParams(dset,params):
    for pname in dset.config["globalParams"]:
        if pname in params:
            dset.config["globalParams"][pname] = params[pname]

    print("Ensure collection fragment is not empty...")
    colSize = dset.collection.getSize(query=dset.config["globalParams"]["collectionFilter"])

    if colSize["count"] > 0:
        print("Fragment size: "+str(colSize["count"]))
    else:
        raise ValueError("No items in fragment.")

    return dsetConfigureLevels(dset)

def datasetSetSpecParams(dset,params):
    for pname in dset.getDefaultConfig()["specParams"]:
        if pname in params:
            dset.config["specParams"][pname] = params[pname]
    return True

def datasetSetSplittingParams(dset,params):
    for pname in dset.getDefaultConfig()["splittingParams"]:
        if pname in params:
            dset.config["splittingParams"][pname] = params[pname]
    return True
