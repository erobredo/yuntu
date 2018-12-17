
from abc import abstractmethod, ABCMeta
from core import dataframe
from core.collection import timedAudioCollection
import numpy as np
from tqdm import tqdm


class Datascape(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def initDataFrame(self):
        pass

    @abstractmethod
    def summary(self,groupBy,condition=None):
        pass

class metaSoundscape(Datascape):
    __metaclass__ = ABCMeta

    def __init__(self,name,collection,timeStep=60,freqStep=100,energyTransform=None,preProcess=None,flimits=None,channel=None,n_fft=1024,hop_length=512):
        self.name = name
        self.timeStep = timeStep
        self.freqStep = freqStep
        self.n_fft=n_fft 
        self.hop_length=hop_length
        self.channel = channel
        self.flimits = flimits
        self.collection = collection
        self.preProcess = preProcess
        self.energyTransform = energyTransform

        self.initDataFrame()


    def initDataFrame(self):
        print("Generating soundscape...")
        energySamples = []

        pbar = tqdm(total=self.collection.size)
        for md5 in self.collection.data.keys():
            audio = self.collection.data[md5]
            if self.channel is None:
                channels = range(audio.nchannels)
            else:
                channels = [self.channel]

            for ch in channels:
                energies = energySteps(audio,tstep=self.timeStep,fstep=self.freqStep,energyTransform=self.energyTransform,preProcess=self.preProcess,flimits=self.flimits,channel=ch,n_fft=self.n_fft,hop_length=self.hop_length)
                for e in energies:
                    e["md5"] = md5
                    for mkey in audio.metadata.keys():
                        e[mkey] = audio.metadata[mkey]

                    energySamples.append(e)
            pbar.update(1)

        self.df = dataframe.fromArray(energySamples)



    def summary(self,groupBy,condition=None):
        results = None

        if condition is None:
            if results is None:
                results = self.df.groupby(groupBy).apply(energySummary)
            else:
                results = results.join(self.df.groupby(groupBy).apply(energySummary))
        else:
            if results is None:
                results = self.df[condition].groupby(groupBy).apply(energySummary)
            else:
                results = results.join(self.df[condition].groupby(groupBy).apply(energySummary))

        return results


class cronoSoundscape(metaSoundscape):

    def __init__(self,name,collection,timeStep=60,freqStep=100,energyTransform=None,preProcess=None,flimits=None,channel=None,n_fft=1024,hop_length=512):

        if isinstance(collection,timedAudioCollection):
            super(cronoSoundscape, self).__init__(name,collection,timeStep,freqStep,energyTransform,preProcess,flimits,channel,n_fft,hop_length)
        else:
            raise ValueError("A collection of class 'cronoAudioCollection' must be supplied in order to build a 'cronoSoundscape'.")



def energySteps(audio,tstep,fstep,energyTransform=None,preProcess=None,flimits=None,channel=0,n_fft=1024,hop_length=512,norm=True):
    audio.unset_mask()

    spec, freqs = audio.get_spec(channel,n_fft,hop_length,preProcess)
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

        spec = spec[minFreqIdx:maxFreqIdx,:]
        freqRange = flimits[1]-flimits[0]

    fbins,tbins = spec.shape

    tstep_ = int(round(tbins*(float(tstep)/duration)))
    fstep_ = int(round(fbins*(float(fstep)/freqRange)))


    ntsteps = int(round(float(tbins)/tstep_))
    nfsteps = int(round(float(fbins)/fstep_))

    larger = False
    if tstep < duration:
        splitsTime = [i*tstep_+tstep_ for i in range(ntsteps-1)]
        timeSplit = np.split(spec,splitsTime,axis=1)
    else:
        timeSplit = [spec]
        larger = True


    splitsFreq = [i*fstep_+fstep_ for i in range(nfsteps-1)]


    energies = []

    for i in range(len(timeSplit)):
        e = timeSplit[i]

        if energyTransform is not None:
            e = energyTransform(e)

        if norm:
            maxe = np.amax(e)
            if maxe > 0:
                e = e/maxe

        e = np.sum(e,axis=1)
        e = np.array(np.split(e,splitsFreq,axis=0))
        e = np.sum(e,axis=1)

        if not larger:
            start = i*tstep
            stop = i*tstep+tstep

            if i == len(timeSplit)-1:
                stop = duration
        else:
            start = 0
            stop = duration

        energies.append({"energy":e,"start":start,"stop":stop,"channel":channel})

    audio.clear_media()

    return energies


def energySummary(x):
    d = {}
    w = (x["stop"]-x["start"])/((x["stop"]-x["start"]).sum())

    d["nsamples"] = x["md5"].count()
    d["energy_sum"] = x["energy"].sum()
    d["energy_mean"] = (x['energy']*w).sum()
    d["energy_var"] = (((x['energy']-d["energy_mean"])**2)*w).sum()
    d["energy_std"] = np.sqrt(d["energy_var"])


    
    return dataframe.makeSeries(d,['nsamples','energy_mean','energy_std','energy_var','energy_sum'])





