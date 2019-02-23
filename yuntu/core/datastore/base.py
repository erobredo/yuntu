from abc import abstractmethod, ABCMeta
import yuntu.core.datastore.methods as dsMethods

class metaDatastore(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getSpec(self):
        pass

    @abstractmethod
    def getConf(self):
        pass

    @abstractmethod
    def getMetadata(self):
        pass

    @abstractmethod
    def getHash(self):
        pass

class simpleDatastore(metaDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,inputSpec):
        self.inputSpec = inputSpec

    def getType(self):
        return dsMethods.datastoreGetType(self)

    def getSpec(self):
        return dsMethods.datastoreGetSpec(self)

    def getConf(self):
        return dsMethods.datastoreGetConf(self)

    def getMetadata(self):
        return dsMethods.datastoreGetMetadata(self)

    def getHash(self):
        return dsMethods.datastoreGetHash(self)

    def getData(self):
        pass

class activeDatastore(simpleDatastore):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getData(self):
        pass

class directDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,dataArr):
        self.inputSpec = {
                        "type":"direct",
                        "conf":{
                            "host":None,
                            "datastore":None,
                            "target":None,
                            "filter":None,
                            "fields":None,
                            "ukey":None
                            },
                        "metadata":{
                            "description":"Direct input as dict array."
                            }
                        }
        self.dataArr = dataArr

    def getData(self):
        return dsMethods.datastoreDirectGetData(self)

class mongoDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def getData(self):
        return dsMethods.datastoreMongoGetData(self)
