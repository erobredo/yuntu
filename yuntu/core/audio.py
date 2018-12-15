from abc import abstractmethod,ABCMeta
import utils

class Media(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_media_info(self):
        pass

    @abstractmethod
    def read_media(self):
        pass

    @abstractmethod
    def write_media(self,path):
        pass

class Audio(Media):
    __metaclass__ = ABCMeta

    def __init__(self, config, read_sr=None, from_config=False, metadata={}):
        self.read_sr = read_sr
        self.config = config
        self.timeexp = self.config["timeexp"]
        self.path = self.config["path"]
        self.metadata = metadata

        if from_config:
            self.load_basic_info()
        else:
            self.read_basic_info()
            

    def load_basic_info(self):
        self.original_sr = self.config["sr"]
        self.nchannels = self.config["nchannels"]
        self.sampwidth = self.config["sampwidth"]
        self.length = self.config["length"]
        self.md5 = self.config["md5"]
        self.duration = self.config["duration"]
        self.filesize = self.config["filesize"]
        self.sr = self.original_sr
        self.mask = None
        self.signal = None

    def read_basic_info(self):
        self.original_sr,self.nchannels,self.sampwidth,self.length, self.filesize = utils.read_info(self.path)
        self.duration = (float(self.length)/float(self.original_sr))/self.timeexp
        self.sr = self.original_sr
        self.mask = None
        self.signal = None

    def read_media(self):
        offset = 0.0
        duration = None

        if self.mask is not None:
            offset = self.mask[0]
            duration = self.mask[1]-self.mask[0]

        self.signal, self.sr = utils.read(self.path,self.read_sr,offset,duration)

    def clear_media(self):
        self.signal = None
        self.sr = self.original_sr

    def set_read_sr(self,sr):
        self.read_sr = sr
        self.clear_media()
    
    def set_metadata(self,metadata):
        self.metadata = metadata
        
    def set_mask(self,startTime,endTime):
        self.mask = [startTime/self.timeexp,endTime/self.timeexp]
        self.read_media()

    def unset_mask(self):
        self.mask = None
        self.clear_media()

    def get_media_info(self):
        info = {}
        info["path"] = self.path
        info["filesize"] = self.filesize
        info["md5"] = self.md5
        info["timeexp"] = self.timeexp
        info["samplerate"] = self.original_sr
        info["sampwidth"] = self.sampwidth
        info["length"] = self.length
        info["nchannels"] = self.nchannels
        info["duration"] = self.duration

        return info

    def get_signal(self):
        if self.signal is None:
            self.read_media()

        return self.signal


    def get_zcr(self,channel=0,frame_length=1024,hop_length=512):
        if channel > self.nchannels -1:
            raise ValueError("Channel outside range.")

        sig = self.get_signal()

        sig = utils.sigChannel(sig,channel,self.nchannels)

        
        return utils.zero_crossing_rate(sig,frame_length,hop_length)       

    def get_spec(self, channel=0, n_fft=1024, hop_length=512):
        if channel > self.nchannels -1:
            raise ValueError("Channel outside range.")

        sig = self.get_signal()

        sig = utils.sigChannel(sig,channel,self.nchannels)


        return utils.spectrogram(sig,n_fft=n_fft,hop_length=hop_length), utils.spec_frequencies(self.sr,n_fft)


    def get_mfcc(self,channel=0,sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho'):
        if channel > self.nchannels -1:
            raise ValueError("Channel outside range.")

        sig = self.get_signal()
        sig = utils.sigChannel(sig,channel,self.nchannels)

        return utils.mfcc(sig,sr=self.sr, S=None, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm)

    def write_media(self,path,media_format="wav",sr=None):
        if media_format in ["wav","flac","ogg"]:
            sig = self.get_signal()
            out_sr = self.sr

            if sr is not None:
                out_sr = sr

            utils.write(path,sig,out_sr,self.nchannels,media_format)

            return path
        else:
            raise ValueError("Writing to '"+media_format+"' is not supported yet.")

    def plot_spec(self,ax,channel=0,n_fft=1024,hop_length=512):
        spec, freqs = self.get_spec(channel=channel,n_fft=n_fft,hop_length=hop_length)

        return utils.plot_power_spec(spec,ax,self.sr)

    def plot_waveform(self,ax,channel=0,wtype="simple"):
        sig = self.get_signal()
        sig = utils.sigChannel(sig,channel,self.nchannels)
        return utils.plot_waveform(sig,self.sr,ax,wtype=wtype)



