from abc import abstractmethod, ABCMeta
import dataframe


class mediaCollection(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def initDataframe(self,index):
        pass

    @abstractmethod
    def getColType(self):
        pass

    @abstractmethod
    def getMedia(self,index):
        pass

    @abstractmethod
    def dropMedia(self,index):
        pass

    @abstractmethod
    def upsertMedia(self,arr):
        pass

    @abstractmethod
    def dump(self,basePath):
        pass

    @abstractmethod
    def materialize(self,basePath):
        pass

    @abstractmethod
    def load(self,basePath):
        pass



class audioCollection(mediaCollection):
    __metaclass__ = ABCMeta

    def __init__(self,name="",data=None,sr=None,metadataParse=None,basePath=None):
        self.name = name
        self.data = data
        self.metadataParse = metadataParse
        self.basePath = basePath
        self.size = 0

        if self.basePath is not None:
            self.load(basePath)
        else:
            self.initDataframe()
            self.setCollectionSamplerate(sr)


    def initDataframe(self):
        if self.data is not None:
            self.data, self.df = dataframe.fromAudioArray(self.data,self.metadataParse)
        else:
            raise ValueError("No data supplied.")

        self.size = len(self.data.keys())

    def setCollectionSamplerate(self,sr):
        if sr is not None:
            self.sr = sr
            for key in self.data:
                self.data[key].set_read_sr(self.sr)

    def setSamplingResolution(self,timeStep=None,freqStep=None):
        if timeStep is not None:
            self.timeStep = timeStep
        if freqStep is not None:
            self.freqStep = freqStep

    def getColType(self):
        return "basic"

    def getMedia(self,condition=None):
        if condition is None:
            indArr =  list(self.df["md5"])
        else:
            indArr = list(self.df[condition]["md5"])

        return [self.data[key] for key in indArr]

    def getConfig(self):
        config = {}
        config["sr"] = self.sr
        config["name"] = self.name
        config["colType"] = self.getColType()

        return config

    def dropMedia(self,condition=None):
        self.df, keyArr = dataframe.dropData(self.df,condition)
        for key in keyArr:
            del self.data[key]

    def upsertMedia(self,arr,metadataParse=None):
        upserted, self.df = dataframe.upsertData(self.df,arr,self.metadata,metadataParse)
        for key in upserted:
            self.data[key] = upserted[key]

    def load(self,basePath):
        self.data, self.df, self.size, config = dataframe.loadDataFrame(basePath)
        self.name = config["name"]
        self.setCollectionSamplerate(config["sr"])

    def dump(self,basePath,overwrite=False):
        return dataframe.dumpMetadata(self.data,self.df,basePath,self.getConfig(),overwrite)
        
    def materialize(self,basePath,media_format="wav",overwrite=False):
        return dataframe.dumpDataFrame(self.data,self.df,basePath,self.getConfig(),media_format,overwrite)


class timedAudioCollection(audioCollection):

    def initDataframe(self):
        if self.data is not None:
            self.data, self.df = dataframe.fromAudioArray(self.data,self.metadataParse,validateTime=True)
        else:
            raise ValueError("No data supplied.")

        self.size = len(self.data.keys())

    def getColType(self):
        return "timed"

    def upsertMedia(self,arr,metadataParse=None):
        upserted, self.df = dataframe.upsertData(self.df,arr,self.metadata,metadataParse,validateTime=True)
        for key in upserted:
            self.data[key] = upserted[key]

    def getMediaByTime(self,timeInterval):
        t0 = dataframe.timeToSeconds(timeInterval[0])
        t1 = dataframe.timeToSeconds(timeInterval[1])
        condition = (self.df["absTime"] >= t0)&(self.df["absTime"] <= t1)
        return getMedia(condition)

def fromArray(arr,pathField="path",timeexpField=None,name="",sr=None,metadataParse=None,colType="basic"):
    audioArr = dataframe.arr2audio(arr,pathField,timeexpField)
    if colType == "basic":
        return audioCollection(name,audioArr,sr,metadataParse)
    elif colType == "timed":
        return timedAudioCollection(name,audioArr,sr,metadataParse)
    else:
        raise ValueError("Value of 'colType' must be in ['basic','timed'].")

def fromCsv(filePath,delimiter=",",pathField="path",timeexpField=None,name="",sr=None,metadataParse=None,colType="basic"):
    audioArr = dataframe.csv2audio(filePath,delimiter,pathField,timeexpField)
    if colType == "basic":
        return audioCollection(name,audioArr,sr,metadataParse)
    elif colType == "timed":
        return timedAudioCollection(name,audioArr,sr,metadataParse)
    else:
        raise ValueError("Value of 'colType' must be in ['basic','timed'].")
    
def fromMongo(dbConfig,pathField="path",timeexpField=None,name="",sr=None,metadataParse=None,colType="basic"):
    audioArr = dataframe.mongo2audio(dbConfig,pathField,timeexpField)
    if colType == "basic":
        return audioCollection(name,audioArr,sr,metadataParse)
    elif colType == "timed":
        return timedAudioCollection(name,audioArr,sr,metadataParse)
    else:
        raise ValueError("Value of 'colType' must be in ['basic','timed'].")
    
#def fromSqlite(dbConfig,pathField="path",timeexpField=None,name="",sr=None,metadataParse=None):
#    audioArr = dataframe.sqlite2audio(dbConfig,pathField,timeexpField)
#    return audioCollection(name,audioArr,sr,metadataParse)

#def fromPg(dbConfig,pathField="path",timeexpField=None,name="",sr=None,metadataParse=None):
#    audioArr = dataframe.pg2audio(dbConfig,pathField,timeexpField)
#    return audioCollection(name,audioArr,sr,metadataParse)

def load(basePath):
    config = dataframe.loadConfig(basePath)
    if config["colType"] == "basic":
        return audioCollection(basePath=basePath)
    elif config["colType"] == "timed":
        return timedAudioCollection(basePath=basePath)
    else:
        raise ValueError("Wrong collection config.")

def dump(col,basePath,overwrite=False):
    return col.dump(basePath,overwrite)

def materialize(col,basePath,media_format="wav",overwrite=False):
    return col.materialize(basePath,media_format,overwrite)






