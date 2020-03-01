from abc import abstractmethod, ABCMeta
import yuntu.soundscape.methods as scMethods

class metaSoundscape(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def setConfig(self,ncores):
        pass

    @abstractmethod
    def setGlobalParams(self,globalParams):
        pass

    @abstractmethod
    def setEnergyParams(self,energyParams):
        pass

    @abstractmethod
    def setSamplingParams(self,samplingParams):
        pass

    @abstractmethod
    def setAcousticIndicesParams(self,indexParams):
        pass

    @abstractmethod
    def setSamplingParams(self,samplingParams):
        pass

    @abstractmethod
    def getType(self):
        pass

    @abstractmethod
    def getDefaultConfig(self):
        pass

    @abstractmethod
    def getGroups(self):
        pass

    @abstractmethod
    def getFragment(self,groupName):
        pass

    # @abstractmethod
    # def getEnergy(self,groupName):
    #     pass

    @abstractmethod
    def loadGraph(self):
        pass


class simpleSoundscape(metaSoundscape):
    __metaclass__ = ABCMeta

    def __init__(self,name,collection=None,dirPath="",config=None,metadata=None,client=None,overwrite=False):
        self.name = name
        self.dirPath = dirPath
        self.collection = collection
        self.client = client
        self.config = self.getDefaultConfig()
        self.groupFilters = {"all":None}
        self.outs = []
        self.overwrite = overwrite
        self.info =  {"name":self.name,"dirPath":dirPath,"collection":None,"creation":None,"modification":None,"type":"simpleSoundscape","metadata":metadata}
        self.splits = None
        self.graph = {}


        doBuild = False
        if scMethods.soundscapeExists(self):
            if not self.overwrite:
                print("Loading soundscape...")
                if scMethods.soundscapeLoad(self):
                    self.setConfig(config)
            else:
                doBuild = True
        else:
            doBuild = True

        if doBuild:
            if collection is None:
                raise ValueError("Collection must be explicit on soundscapes creation (create a collection and pass as parameter)")
            print("Building new soundscape...")
            if scMethods.soundscapeBuild(self):
                self.setConfig(config)

        self.loadGraph()


    def setConfig(self,config):
        return scMethods.soundscapeSetConfig(self,config)

    def setNode(self,name,opDef,isOutput=False):
        return scMethods.soundscapeSetNode(self,name,opDef,isOutput)

    def getType(self):
        return "simpleSoundscape"

    def getDefaultConfig(self):
        config = {
            "globalParams" : {
                "multiThread" : False,
                "collectionFilter" : None,
                "groupingFields" : None,
                "groupingTypes" : None,
                "npartitions" : 20
            },
            "energyParams" : {
                "channels" : "__all__",
                "n_fft":1024,
                "hop_length":512,
                "tStep" : 60,
                "fBins" : 10,
                "fLimits" : [0,10000],
                "readSr" : None,
                "transformations" : {
                    "aggr":{
                        "method":"numpy.sum"
                    }
                },
            },
            "indexParams" : {
                "globals" : None,
                "configs" : []
            },
            "samplingParams": {
                "size" : -1,
                "replace" : False
            },
            "datasetParams" : {
                "channels" : "__all__",
                "n_fft":1024,
                "hop_length":512,
                "tStep" : 60,
                "fBins" : 10,
                "fLimits" : [0,10000],
                "readSr" : None,
                "transformations": None
            }
        }
        return config

    def getConfig(self):
        return scMethods.soundscapeGetConfig(self)

    # def getEnergy(self,group=None,compute=True):
    #     return scMethods.soundscapeGetEnergy(self,group,compute)

    def getSplits(self,group=None,compute=True,baseName=""):
        return scMethods.soundscapeGetSplits(self,group,compute,baseName=baseName)

    def getSample(self,group=None,compute=True,baseName=""):
        return scMethods.soundscapeGetSample(self,group,compute,baseName=baseName)

    def getNode(self,nodeName,group=None,compute=True,overwrite=False,baseName=""):
        return scMethods.soundscapeGetNode(self,nodeName=nodeName,group=group,compute=compute,baseName=baseName)

    def getGroups(self):
        return scMethods.soundscapeGetGroups(self)

    def getCounts(self):
        return scMethods.soundscapeGetCounts(self)

    def groupExists(self,group):
        return scMethods.soundscapeGroupExists(self,group)

    def loadGraph(self):
        return scMethods.soundscapeLoadGraph(self)

    def setClient(self,client):
        return scMethods.soundscapeSetClient(self,client)

    def appendOp(self,op):
        pass

    def compute(self):
        return scMethods.soundscapeCompute(self)

    def clear(self):
        return scMethods.soundscapeClear(self)


class cronoSoundscape(simpleSoundscape):
    __metaclass__ = ABCMeta

    def __init__(self,name,collection=None,dirPath="",config=None,metadata=None,client=None,overwrite=False):

        super(cronoSoundscape,self).__init__(**{"name":name,"collection":collection,"dirPath":dirPath,"config":config,"metadata":metadata,"client":client,"overwrite":overwrite})

    def getType(self):
        return "cronoSoundscape"

    def getDefaultConfig(self):
        config = {
            "globalParams" : {
                "multiThread" : False,
                "collectionFilter" : None,
                "groupingFields" : None,
                "groupingTypes" : None,
                "npartitions" : 20,
                "writeSpecs" : True
            },
            "energyParams" : {
                "channels" : "__all__",
                "n_fft":1024,
                "hop_length":512,
                "tStep" : 60,
                "fBins" : 10,
                "fLimits" : [0,10000],
                "readSr" : None,
                "transformations" : {
                    "aggr":{
                        "method":"numpy.sum"
                    }
                },
            },
            "samplingParams": {
                "size" : -1,
                "replace" : False
            },
            "indexParams" : {
                "globals" : None,
                "configs" : []
            },
            "datasetParams" : {
                "channels" : "__all__",
                "n_fft":1024,
                "hop_length":512,
                "tStep" : 60,
                "fBins" : 10,
                "fLimits" : [0,10000],
                "readSr" : None,
                "transformations": None
            },
            "cronoParams" : {
                "startDate" : None,
                "timeZone" : None,
                "timeFormat" : '%d-%m-%Y %H:%M:%S',
                "unit" : 3600,
                "modulo" : 24
            }
        }
        return config

    def setCronoParams(self,cronoParams):
        return scMethods.soundscapeSetCronoParams(self,cronoParams)
    
    def plot(self,varnames=[],orids=[],splits=None,group=None,plot_type="concat",viewTz=None,plotTimeFormat="%H:%M",nlabels=21,xlabel="T",ylabel="F (kHz)",xlabels=[],ylabels=[],cmap="terrain",interpolation="sinc",figsize=(20,10),path=None,baseName="",timeLimits=None,freqLimits=None):
        return scMethods.soundscapeCronoPlot(self,varnames=varnames,orids=orids,splits=splits,group=group,plot_type=plot_type,viewTz=viewTz,plotTimeFormat=plotTimeFormat,nlabels=nlabels,xlabel=xlabel,ylabel=ylabel,xlabels=xlabels,ylabels=ylabels,cmap=cmap,interpolation=interpolation,figsize=figsize,path=path,baseName=baseName,timeLimits=timeLimits,freqLimits=freqLimits)







