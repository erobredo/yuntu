import yuntu.core.audio.utils as auUtils


def audioLoadBasicInfo(au):
    au.originalSr = au.config["samplerate"]
    au.nchannels = au.config["nchannels"]
    au.sampwidth = au.config["sampwidth"]
    au.length = au.config["length"]
    au.md5 = au.config["md5"]
    au.duration = au.config["duration"]
    au.filesize = au.config["filesize"]
    au.sr = au.originalSr
    au.mask = None
    au.signal = None

    return True

def audioReadBasicInfo(au):
    au.originalSr,au.nchannels,au.sampwidth,au.length, au.filesize = auUtils.readInfo(au.path)
    au.duration = (float(au.length)/float(au.originalSr))/au.timeexp
    au.sr = au.originalSr
    au.mask = None
    au.signal = None

    return True

def audioListen(au):
    from IPython.display import Audio
    return Audio(data=au.getSignal(), rate=au.sr)

def audioReadMedia(au):
    offset = 0.0
    duration = None

    if au.mask is not None:
        offset = au.mask[0]
        duration = au.mask[1]-au.mask[0]

    au.signal, au.sr = auUtils.read(au.path,au.readSr,offset,duration)

    return True

def audioClearMedia(au):
    au.signal = None
    au.sr = au.originalSr

    return True


def audioSetReadSr(au,sr):
    au.readSr = sr
    au.clearMedia()

    return True

def audioSetMetadata(au,metadata):
    au.metadata = metadata

    return True

def audioSetMask(au,startTime,endTime):
    au.mask = [startTime/au.timeexp,endTime/au.timeexp]
    au.readMedia()

    return True

def audioUnsetMask(au):
    au.mask = None
    au.clearMedia()

    return True

def audioGetMediaInfo(au):
    info = {}
    info["path"] = au.path
    info["filesize"] = au.filesize
    info["timeexp"] = au.timeexp
    info["samplerate"] = au.originalSr
    info["sampwidth"] = au.sampwidth
    info["length"] = au.length
    info["nchannels"] = au.nchannels
    info["duration"] = au.duration

    return info

def audioGetSignal(au,preProcess=None):
    if au.signal is None:
        au.readMedia()
        if preProcess is not None:
            if isinstance(preProcess,dict):
                au.signal = preProcess["transform"](au.signal,**preProcess["kwargs"])
            else:
                au.signal = preProcess(au.signal)

    return au.signal


def audioGetZcr(au,channel=0,frame_length=1024,hop_length=512,preProcess=None):
    if channel > au.nchannels -1:
        raise ValueError("Channel outside range.")

    sig = au.getSignal(preProcess)

    sig = auUtils.sigChannel(sig,channel,au.nchannels)

    
    return auUtils.zeroCrossingRate(sig,frame_length,hop_length)       

def audioGetSpec(au, channel=None, n_fft=1024, hop_length=512,preProcess=None):
    if channel is None:
        sig = auUtils.channelMean(au.getSignal(preProcess))
    else:
        if channel > au.nchannels -1:
            raise ValueError("Channel outside range.")
            
        sig = au.getSignal(preProcess)
        sig = auUtils.sigChannel(au.getSignal(preProcess),channel,au.nchannels)

    

    return auUtils.spectrogram(sig,n_fft=n_fft,hop_length=hop_length), auUtils.specFrequencies(au.sr,n_fft)


def audioGetMfcc(au,channel=0,sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho',preProcess=None):
    if channel > au.nchannels -1:
        raise ValueError("Channel outside range.")

    sig = au.getSignal(preProcess)
    sig = auUtils.sigChannel(sig,channel,au.nchannels)

    return auUtils.mfcc(sig,sr=au.sr, S=None, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm)

def audioWriteMedia(au,path,media_format="wav",sr=None):
    if media_format in ["wav","flac","ogg"]:
        sig = au.getSignal()
        out_sr = au.sr

        if sr is not None:
            out_sr = sr

        auUtils.write(path,sig,out_sr,au.nchannels,media_format)

        return path
    else:
        raise ValueError("Writing to '"+media_format+"' is not supported yet.")

def audioWriteChunks(au,basePath,chop,thresh=1,media_format="wav",sr=None):
    duration = au.duration
    if sr is not None:
        audioSetReadSr(au,sr)
        
    results = []
    if chop is None:
        out = {"tlimits":[0,duration],"number":0}
        out["path"] = audioWriteMedia(au,basePath+"."+media_format,media_format)
        results = [out]
    elif duration < chop:
        out = {"tlimits":[0,duration],"number":0}
        out["path"] = audioWriteMedia(au,basePath+"."+media_format,media_format)
        results = [out]
    else:
        fchunks = duration/chop
        if fchunks <= thresh:
            out = {"tlimits":[0,duration],"number":0}
            out["path"] = audioWriteMedia(au,basePath+"."+media_format,media_format)
            results = [out]
        else:
            nchunks = int(fchunks)
            residual = fchunks-nchunks

            if residual >= 0.5:
                nchunks += 1

            tlimits = [(float(chop*x),float(min(chop*(x+1),duration))) for x in range(nchunks)]
            for i in range(len(tlimits)):
                au.setMask(tlimits[i][0],tlimits[i][1])
                out = {"tlimits":tlimits[i],"number":i}
                out["path"] = audioWriteMedia(au,basePath+"_chunk_"+str(i)+"."+media_format,media_format)
                results.append(out)
                au.unsetMask()

    return results



