from abc import abstractmethod, ABCMeta
import dataframe
from soundscape import Soundscape

class mediaCollection(object):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def name(self)
        pass

    @property
    @abstractmethod
    def dataFrame(self)
        pass

    @abstractmethod
    def initDataframe(self,index):
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



class audioCollection(mediaCollection):
    __metaclass__ = ABCMeta

    def __init__(self,name,data=None,mongoDict=None):
        self.timeStep = timeStep
        self.freqStep = freqStep
        self.name = name
        self.data = data
        self.mongoDict = mongoDict

        self.initDataframe()

    def initDataframe(self):
        if self.data is not None:
            self.data, self.dataFrame = dataframe.fromAudioArray(self.data)
        elif self.mongoDict is not None:
            self.data, self.dataFrame = dataframe.fromAudioMongoQuery(self.mongoDict)
        else:
            raise ValueError("Variable 'data' is None but no other data source supplied.")

    def getDataFrame(self,condition=True):
        return self.dataFrame[condition]

    def setSamplingResolution(self,timeStep=None,freqStep=None):
        if timeStep is not None:
            self.timeStep = timeStep
        if freqStep is not None:
            self.freqStep = freqStep


    def getMedia(self,condition=True):
        indArr = list(self.dataFrame[condition]["akey"])
        return [self.data[key] for key in keyArr]

    def dropMedia(self,condition=True):
        self.dataFrame, keyArr = dataframe.dropData(self.dataFrame,condition)
        for key in keyArr:
            del self.data[key]

    def upsertMedia(self,arr):
        upserted, self.dataFrame = dataframe.upsertData(self.dataFrame,arr)
        for key in upserted:
            self.data[key] = upserted[key]

    def getSoundscape(self,timeStep=60, freqStep=100, flimits=None, channel=None, n_fft=1024, hop_length=512):
        return Soundscape(self.name+"_soundscape",self.data,timeStep,freqStep,flimits,channel,n_fft,hop_length)








