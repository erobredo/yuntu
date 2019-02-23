from abc import abstractmethod, ABCMeta
import yuntu.collection.methods as colMethods

class mediaCollection(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getMedia(self,query):
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

class audioCollection(mediaCollection):
    __metaclass__ = ABCMeta

    def __init__(self,name,dirPath="",metadata=None,overwrite=False):
        self.name = name
        self.dirPath = dirPath
        self.colPath = colMethods.collectionPath(self)
        self.info = {"name":self.name,"creation":None,"modification":None,"connPath":None,"type":"audioCollection","metadata":metadata}
        self.db = None

        if colMethods.collectionExists(self):
            if overwrite:
                print("Overwriting previous collection...")
                colMethods.collectionDrop(self)
                self.build()
            else:
                print("Loading previous collection...")
                self.load()
        else:
            self.build()

    def build(self):
        return colMethods.collectionBuild(self)

    def load(self):
        return colMethods.collectionLoad(self)

    def getMedia(self,where=None,query=None,iterate=True):
        return colMethods.collectionQuery(self,where,query,iterate)

    def insertMedia(self,input,parseSeq=None):
        return colMethods.collectionInsert(self,input,parseSeq)

    def dropMedia(self,where=None,query=None):
        return colMethods.collectionDropMedia(self,where,query)

    def pullDatastore(self,dsDict,parseSeq=None):
        return colMethods.collectionPullDatastore(self,dsDict,parseSeq)

    def transformMetadata(self,parseSeq,id=None,where=None,query=None,operation="append"):
        return colMethods.collectionTransform(self,parseSeq,id,where,query,operation)

    def dump(self,dirPath,overwrite=False):
        return colMethods.collectionDump(self,dirPath,overwrite)
        
    def materialize(self,dirPath=None,media_format="wav",overwrite=False):
        return colMethods.collectionMaterialize(self,dirPath,media_format,overwrite)


