from abc import abstractmethod,ABCMeta
import yuntu.core.audio.methods as auMethods

class Media(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getMediaInfo(self):
        pass

    @abstractmethod
    def readMedia(self):
        pass

    @abstractmethod
    def writeMedia(self,path):
        pass

class Audio(Media):
    __metaclass__ = ABCMeta

    def __init__(self, config, readSr=None, fromConfig=False, metadata={}):
        self.readSr = readSr
        self.config = config
        self.timeexp = self.config["timeexp"]
        self.path = self.config["path"]
        self.metadata = metadata

        if fromConfig:
            self.loadBasicInfo()
        else:
            self.readBasicInfo()
            
    def loadBasicInfo(self):
        return auMethods.audioLoadBasicInfo(self)

    def readBasicInfo(self):
        return auMethods.audioReadBasicInfo(self)

    def readMedia(self):
        return auMethods.audioReadMedia(self)

    def clearMedia(self):
        return auMethods.audioClearMedia(self)

    def setReadSr(self,sr):
        return auMethods.audioSetReadSr(self,sr)
    
    def setMetadata(self,metadata):
        return auMethods.audioSetMetadata(self,metadata)
        
    def setMask(self,startTime,endTime):
        return auMethods.audioSetMask(self,startTime,endTime)

    def unsetMask(self):
        return auMethods.audioUnsetMask(self)

    def getMediaInfo(self):
        return auMethods.audioGetMediaInfo(self)

    def getSignal(self,preProcess=None):
        return auMethods.audioGetSignal(self,preProcess)

    def getZcr(self,channel=0,frame_length=1024,hop_length=512,preProcess=None):
        return auMethods.audioGetZcr(self,channel,frame_length,hop_length,preProcess)  

    def getSpec(self, channel=0, n_fft=1024, hop_length=512,preProcess=None):
        return auMethods.audioGetSpec(self,channel,n_fft,hop_length,preProcess)

    def getMfcc(self,channel=0,sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho',preProcess=None):
        return auMethods.audioGetMfcc(self,channel,sr,S,n_mfcc,dct_type,norm,preProcess)

    def writeMedia(self,path,media_format="wav",sr=None):
        return auMethods.audioWriteMedia(self,path,media_format,sr)

    def writeChunks(self,basePath,chop,thresh,media_format="wav",sr=None):
        return auMethods.audioWriteChunks(self,basePath,chop,thresh,media_format,sr)
    
    def listen(self):
        return auMethods.audioListen(self)






