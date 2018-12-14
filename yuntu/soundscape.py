
import dataframe
import numpy as np

#Functions

def energySteps(audio,tstep,fstep,flimits=None,channel=0,n_fft=1024,hop_length=512,norm=True):
    audio.unset_mask()

    spec, freqs = audio.get_spec(channel,n_fft,hop_length)
    duration = audio.duration
    topFreq = np.amax(freqs)
    fbins = freqs.size

    maxFreqIdx = fbins
    minFreqIdx = 0

    freqRange = topFreq

    if flimits is not None:

        maxFreqIdx = min(int(round(fbins*flimits[1]/freqRange)),fbins)
        minFreqIdx = min(int(round(fbins*flimits[0]/freqRange)),fbins)

        if maxFreqIdx <= minFreqIdx:
            raise ValueError("Wrong frequency limits")

        spec = spec[:,maxFreqIdx:minFreqIdx]
        freqRange = flimits[1]-flimits[0]

    tbins,fbins = spec.shape

    tstep_ = int(round(tbins*(float(tstep)/duration)))
    fstep_ = int(round(fbins*(float(fstep)/freqRange)))

    ntsteps = int(round(float(tbins)/tstep_))
    nfsteps = int(round(float(fbins)/fstep_))

    energies = [{"energy": np.sum(np.concatenate([np.sum(spec[:,f*fstep_:f*fstep_+fstep_],axis=1) for f in range(nfsteps) if f*fstep_+fstep_ < fbins],axis=1)[i*tstep_:i*tstep_+tstep_,:],axis=0),"start":i*tstep,"stop":i*tstep+tstep,"channel":channel} for i in range(ntsteps) if i*tstep_+tstep_ < tbins]

    if norm:
        for i in range(len(energies)):
            e = energies[i]["energy"]
            maxe = np.amax(e)
            energies[i]["energy"] = e/maxe

    audio.clear_media()

    return energies


def shannon(vec):
    sum_vec = np.sum(vec)
    result = None

    if sum_vec > 0:
        norm_vec = vec/sum_vec
        result = -(np.sum(norm_vec*np.log(norm_vec)))

    return result


def energyMean(sdf,groupBy,condition=True):
    df = dataframe.vectorGroupBy(sdf,target="energy",agg_func=np.mean,groupBy=groupBy,condition=condition)
    return df

def energySum(sdf,groupBy,condition=True):
    df = dataframe.vectorGroupBy(sdf,target="energy",agg_func=np.sum,groupBy=groupBy,condition=condition)
    return df

def energyVar(sdf,groupBy,condition=True):
    df = dataframe.vectorGroupBy(sdf,target="energy",agg_func=np.var,groupBy=groupBy,condition=condition)
    return df

def energyStd(sdf,groupBy,condition=True):
    df = dataframe.vectorGroupBy(sdf,target="energy",agg_func=np.std,groupBy=groupBy,condition=condition)
    return df

def ADI(sdf,condition):
    df = dataframe.vectorApply(sdf,target="energy",map_func=shannon,condition=condition)
    return df

#def AEI(sdf,condition):
#    df = dataframe.vectorApply(sdf,target="energy",func=gini,condition=condition)
#    return df

#classes
class Soundscape(object):
    def __init__(self,name,data,timeStep=60,freqStep=100,flimits=None, channel=None, n_fft=1024, hop_length=512):
        self.name = name
        self.data = data
        self.timeStep = timeStep
        self.freqStep = freqStep
        self.n_fft=n_ftt 
        self.hop_length=hop_length
        self.channel = channel

        self.initDataFrame()

    def initDataFrame(self):
        if isinstance(self.data, list):
            data = {}
            for i in range(len(self.data)):
                data[i] = self.data[i]

        self.data = data

        energySamples = []
        for akey in self.data.keys():
            audio = self.data[akey]

            if self.channel is None:
                channels = range(audio.nchannels)
            else:
                channels = [self.channel]

            for ch in channels:
                energies = energySteps(audio,tstep=self.timeStep,fstep=self.freqStep,flimits=self.flimits,channel=ch,n_fft=self.n_fft,hop_length=self.hop_length)
                for e in energies:
                    e["akey"] = akey
                    for mkey in audio.metadata.keys():
                        e[mkey] = audio.metadata[mkey]

                    energySamples.append(e)

        self.dataFrame = dataframe.fromArray(energySamples)



    def getDataFrame(self,condition=True):
        return self.dataFrame[condition]

    def sum(self,groupBy,condition=True):
        return energySum(self.dataFrame,groupBy,condition)

    def mean(self,groupBy,condition=True):
        return energyMean(self.dataFrame,groupBy,condition)

    def var(self,groupBy,condition=True):
        return energyVar(self.dataFrame,groupBy,condition)

    def std(self,groupBy,condition=True):
        return energyStd(self.dataframe,groupBy,condition)







