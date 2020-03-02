from abc import abstractmethod, ABCMeta
import yuntu.dataset.methods as dsetMethods

class metaDataset(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getType(self):
        pass

    @abstractmethod
    def getDefaultConfig(self):
        pass

    @abstractmethod
    def setConfig(self):
        pass

class audioDataset(object):
    __metaclass__ = ABCMeta

    def __init__(self,name,collection=None,dirPath="",config=None,metadata=None,client=None,overwrite=False):
        self.name = name
        self.dirPath = dirPath
        self.collection = collection
        self.client = client
        self.config = self.getDefaultConfig()
        self.overwrite = overwrite
        self.info =  {"name":self.name,"dirPath":dirPath,"collection":None,"creation":None,"modification":None,"type":"audioDataset","metadata":metadata}
        self.graph = {}

        doBuild = False
        if dsetMethods.datasetExists(self):
            if not self.overwrite:
                print("Loading soundscape...")
                if dsetMethods.datasetLoad(self):
                    self.setConfig(config)
            else:
                doBuild = True
        else:
            doBuild = True

        if doBuild:
            if collection is None:
                raise ValueError("Collection must be explicit in dataset creation (create a collection and pass as parameter)")
            print("Building new soundscape...")
            if dsetMethods.datasetBuild(self):
                self.setConfig(config)

        self.loadGraph()

    def getType(self):
        return "audioDataset"

    def setConfig(self,config):
        return dsetMethods.datasetSetConfig(self,config)

    def getDefaultConfig(self):
        config = {
            "globalParams" : {
                "multiThread" : False,
                "collectionFilter" : None,
                "groupingFields" : None,
                "groupingTypes" : None,
                "npartitions" : 20,
                "annotationFabric" : "default"
            },
            "transformationParams" : {
                "annotationFabric" : None,
                "exampleFabric" : None
            },
            "splittingParams": {
                "folds" : 5,
                "splits" : {
                    "train" : 70,
                    "test" : 20,
                    "validation" : 10
                }
            },
        }
        return config
