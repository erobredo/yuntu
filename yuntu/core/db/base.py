from abc import abstractmethod,ABCMeta
import yuntu.core.db.methods as dbMethods

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



class embeddedDb(metaDb):
    __metaclass__ = ABCMeta

    def __init__(self,name,dirPath,overwrite=False):
        self.name = name
        self.dirPath = dirPath
        self.connPath = dbMethods.lDbConnPath(self)
        self.overwrite = overwrite
        self.connection = None

        if dbMethods.lDbExists(self):
            if overwrite:
                print("Overwriting previous database...")
                dbMethods.lDbDrop(self)
                self.build()
            else:
                print("Reading previous database...")
                self.connect()
        else:
            self.build()

    def build(self):
        return dbMethods.lDbCreateStructure(self)

    def insert(self,dataArray,parseSeq=[]):
        return dbMethods.lDbInsert(self,dataArray,parseSeq)

    def connect(self):
        return dbMethods.lDbConnect(self)

    def close(self):
        return dbMethods.lDbClose(self)

    def dump(self,path,overwrite=False):
        return dbMethods.lDbDump(self,path,overwrite)

    def find(self,id=None,query=None):
        return dbMethods.lDbFind(self,id,query)

    def select(self,id=None,where=None):
        return dbMethods.lDbSelect(self,id,where)

    def remove(self,id=None,where=None,query=None):
        return dbMethods.lDbRemove(self,id,query)

    def transform(self,parseSeq,id=None,where=None,query=None,operation="append"):
        wStatement = where
        if query is not None:
            wStatement = dbMethods.lDbParseQuery(query)
        return dbMethods.lDbUpdateParseSeq(self,parseSeq,id,wStatement,operation)


