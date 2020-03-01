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

class activeDatastore(simpleDatastore):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getData(self):
        pass
    
    @abstractmethod
    def getSpecExample(self):
        pass

class directDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,dataArr):
        """
        No spec needed.
        """
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

class audioMothDatastore(activeDatastore):
    __metaclass__ = ABCMeta
    
    def __init__(self,inputSpec):
        """
        inputSpec ~ {
            "type":"audioMoth",
            "conf":{
                "dataDir":"/foo/"
            },
            "metadata":{
                "description":"Audiomoth datastore."
            }
        }
        """
        self.inputSpec = inputSpec
        self.inputSpec["type"] = "audioMoth"
        
    def getData(self):
        return dsMethods.datastoreAudioMothGetData(self)

class mongodbDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,inputSpec):
        """
        inputSpec ~ {
            "type":"mongodb",
            "conf":{
                "host" : "mongodb://<ip>:<port>/",
                "datastore" : <db_name>,
                "target" : <collection_name>,
                "filter" : <mongo_filter>,
                "fields" : <foo1:1,foo2:1>,
                "ukey" : "_id"
            },
            "metadata": {
                "description": "Datastore induced by mongodb query",
            }
        }
        """
        self.inputSpec = inputSpec
        self.inputSpec["type"] = "mongodb"
        
    def getData(self):
        return dsMethods.datastoreMongodbGetData(self)

class postgresqlDatastore(activeDatastore):
    __metaclass__ = ABCMeta
    
    def __init__(self,inputSpec):
        """
        inputSpec ~ {
            "type":"postgresql",
            "conf":{"dbname='snmb_development' user='postgres' host='coati.conabio.gob.mx' password='000999000.'"
                "host" : <ip or host_name>,
                "user" : <user_name>,
                "password" : <user_password>
                "datastore" : <db_name>,
                "target" : <select_statement>,
                "ukey" : <id_from_select>
            },
            "metadata": {
                "description": "Datastore induced by postgresql query",
            }
        }
        """
        self.inputSpec = inputSpec
        self.inputSpec["type"] = "postgresql"
        
    def getData(self):
        return dsMethods.datastorePostgresqlGetData(self)

class csvDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,inputSpec):
        self.inputSpec = inputSpec
        self.inputSpec["type"] = "csv"
        
    def getData(self):
        pass

class xlsDatastore(activeDatastore):
    __metaclass__ = ABCMeta

    def __init__(self,inputSpec):
        self.inputSpec = inputSpec
        self.inputSpec["type"] = "xls"
        
    def getData(self):
        pass