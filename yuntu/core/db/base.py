from abc import abstractmethod,ABCMeta
import yuntu.core.db.methods as dbMethods

#Fit to pep

class metaDb(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def connect(self,path):
        pass

    @abstractmethod
    def dump(self,path):
        pass

    @abstractmethod
    def find(self,id,query):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def getType(self):
        pass

class embeddedDb(metaDb):
    __metaclass__ = ABCMeta

    def __init__(self,name,dirPath,dump=None,overwrite=False):
        self.name = name
        self.dirPath = dirPath
        self.connPath = dbMethods.lDbConnPath(self)
        self.connection = None
        self.parsersDir = None
        self.parsers = None

        if dbMethods.lDbExists(self):
            if overwrite:
                print("Overwriting previous database...")
                dbMethods.lDbDrop(self)
                self.build(dump)
            else:
                print("Reading previous database...")
                self.connect()
        else:
            self.build(dump)

    def getType(self):
        return 'embeddedDb'

    def build(self,dump=None):
        if dump is None:
            return dbMethods.lDbCreateStructure(self)
        else:
            return dbMethods.lDbCreateFromDump(self,dump)

    def insert(self,dataArray,parseSeq=[],timeConf=None):
        return dbMethods.lDbInsert(self,dataArray,parseSeq,timeConf)

    def annotate(self,dataArr):
        return dbMethods.lDbAnnotate(self,dataArr)

    def connect(self):
        return dbMethods.lDbConnect(self)

    def close(self):
        return dbMethods.lDbClose(self)

    def count(self,where=None,query=None,groupby=None):
        return dbMethods.lDbCount(self,where,query)

    def dump(self,path,overwrite=False):
        return dbMethods.lDbDump(self,path,overwrite)

    def find(self,id=None,query=None,table="parsed"):
        return dbMethods.lDbFind(self,id,query,table)

    def select(self,id=None,where=None,freeSt=None,table="parsed"):
        return dbMethods.lDbSelect(self,id,where,freeSt,table)

    def remove(self,id=None,where=None,query=None):
        return dbMethods.lDbRemove(self,id,query)

    def asStatements(self):
        return dbMethods.lDbAsStatements(self)

    def transform(self,parseSeq,id=None,where=None,query=None,operation="append"):
        wStatement = where
        if query is not None:
            wStatement = dbMethods.lDbParseQuery(query)
        return dbMethods.lDbUpdateParseSeq(self,parseSeq,id,wStatement,operation)


class RAMDb(embeddedDb):
    __metaclass__ = ABCMeta
    def __init__(self,name,dump):
        self.name = name
        self.connPath = ':memory:'
        self.connection = None
        self.parsersDir = None
        self.parsers = None
        self.dirPath = None
        self.build(dump)

    def getType(self):
        return 'RAMDb'

    def toDisk(self,dirPath,name=None):
        if name is None:
            name = self.name

        return embeddedDb(name,dirPath,dump=dbMethods.lDbAsStatements(self))
