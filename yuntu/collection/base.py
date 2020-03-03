from abc import abstractmethod, ABCMeta
import yuntu.collection.methods as colMethods

class metaCollection(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getMedia(self,query):
        pass

    @abstractmethod
    def getType(self):
        pass

    @abstractmethod
    def insertMedia(self,dataArray,parseSeq):
        pass

    @abstractmethod
    def dump(self,basePath):
        pass

    @abstractmethod
    def materialize(self,basePath):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def serve(self):
        pass

class simpleCollection(metaCollection):
    __metaclass__ = ABCMeta

    def __init__(self,name,dirPath="",metadata=None,virtual=False,dbDump=None,overwrite=False):
        self.name = name
        self.virtual = virtual
        self.metadata = metadata

        if self.virtual:
            self.dirPath = None
            self.colPath = ":memory:"
            self.info = {"name":self.name,"dirPath":self.dirPath,"creation":None,"modification":None,"connPath":None,"type":"simpleCollection","metadata":metadata}
            self.db = None
            self.virtualInit(dbDump)
        else:
            self.dirPath = dirPath
            self.colPath = colMethods.collectionPath(self)
            self.info = {"name":self.name,"dirPath":dirPath,"creation":None,"modification":None,"connPath":None,"type":"simpleCollection","metadata":metadata}
            self.db = None

            if colMethods.collectionExists(self):
                if overwrite:
                    print("Overwriting previous collection...")
                    colMethods.collectionDrop(self)
                    self.build(dbDump)
                else:
                    print("Loading previous collection...")
                    self.load()
            else:
                self.build(dbDump)

    def virtualInit(self,dbDump):
        colMethods.collectionVirtualInit(self,dbDump)

    def getType(self):
        return "simpleCollection"

    def build(self,dbDump=None):
        return colMethods.collectionBuild(self,dbDump)

    def load(self):
        return colMethods.collectionLoad(self)

    def getMedia(self,orid=None,query=None,iterate=True):
        return colMethods.collectionQuery(self,orid,query,iterate)

    def getMetadata(self,orid=None,query=None,iterate=True):
        return colMethods.collectionGetMetadata(self,orid,query,iterate)
    
    def getAnnotations(self,noteid=None,query=None,iterate=True):
        return colMethods.collectionGetAnnotations(self,noteid,query,iterate)

    def getSignals(self,orid=None,query=None,readSr=None,iterate=True):
        return colMethods.collectionGetSignals(self,orid,query,readSr,iterate)

    def getSpecs(self,orid=None,query=None,readSr=None,n_fft=1024,hop_length=512,iterate=True):
        return colMethods.collectionGetSpecs(self,orid,query,readSr,n_fft,hop_length,iterate)

    def insertMedia(self,input,parseSeq=None):
        return colMethods.collectionInsert(self,input,parseSeq)

    def annotate(self,dataArr):
        return colMethods.collectionAnnotate(self,dataArr)

    def dropMedia(self,where=None,query=None):
        return colMethods.collectionDropMedia(self,where,query)

    def pullDatastore(self,dsDict,parseSeq=None):
        return colMethods.collectionPullDatastore(self,dsDict,parseSeq)

    def transformMetadata(self,parseSeq,id=None,where=None,query=None,operation="append"):
        return colMethods.collectionTransform(self,parseSeq,id,where,query,operation)

    def getSize(self,where=None,query=None):
        return colMethods.collectionGetSize(self,where,query)

    def dump(self,dirPath,overwrite=False):
        return colMethods.collectionDump(self,dirPath,overwrite)

    def materialize(self,dirPath=None,overwrite=False):
        return colMethods.collectionMaterialize(self,dirPath,overwrite)

    def doggyBag(self,dirPath="",outName=None):
        return colMethods.collectionDoggyBag(self,dirPath,outName)

    def toDisk(self,dirPath,overwrite=False):
        return colMethods.collectionToDisk(self,dirPath,overwrite)

    def server(self,port=9797):
        return colMethods.collectionServer(self,port)

class timedCollection(simpleCollection):
    __metaclass__ = ABCMeta

    def __init__(self,name,dirPath="",timeField="datetime",tzField="timezone",timeFormat='%d-%m-%Y %H:%M:%S',metadata=None,virtual=False,dbDump=None,overwrite=False):
        self.timeField = timeField
        self.tzField = tzField
        self.timeFormat = timeFormat

        super(timedCollection,self).__init__(**{"name":name,"dirPath":dirPath,"metadata":metadata,"virtual":virtual,"dbDump":dbDump,"overwrite":overwrite})

    def getType(self):
        return "timedCollection"

    def buildTime(self):
        return colMethods.collectionBuildTime(self)

    def insertMedia(self,input,parseSeq=None):
        return colMethods.collectionTimedInsert(self,input,parseSeq)

    def build(self,dbDump=None):
        if colMethods.collectionBuild(self,dbDump):
            return self.buildTime()

    def load(self):
        if colMethods.collectionLoad(self):
            if self.info["type"] != "timedCollection":
                return self.buildTime()
            else:
                return True
