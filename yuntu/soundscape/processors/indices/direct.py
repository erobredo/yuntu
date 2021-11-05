import numpy as np
from yuntu.soundscape.utils import interpercentile_mean_decibels
from yuntu.soundscape.utils import decile_mod
from yuntu.soundscape.processors.indices.base import AcousticIndex

class TOTAL(AcousticIndex):
    name = 'TOTAL'

    def run(self, array):
        power = np.clip(array, 1e-3, 1e+3)**2
        return 10*np.log10(np.sum(power))


class CORE(AcousticIndex):
    name = 'CORE'

    def run(self, array):
        power = np.clip(array, 1e-3, 1e+3)**2
        ref = 10*np.log10(power)
        _, mod_deciles, perc_ranges = decile_mod(ref)
        return interpercentile_mean_decibels(power, ref, perc_ranges)


class TAIL(AcousticIndex):
    name = 'TAIL'

    def run(self, array):
        power = np.clip(array, 1e-3, 1e+3)**2
        ref = 10*np.log10(power)
        return interpercentile_mean_decibels(power, ref, [(90, 100)])


class INFORMATION(AcousticIndex):
    name = 'INFORMATION'

    def run(self, array):
        power = np.clip(array, 1e-3, 1e+3)**2
        bins = power.shape[0]*power.shape[1]
        normalized_power = power / power.sum()
        log_normalized = np.log2(normalized_power)
        entropy = - (log_normalized * normalized_power).sum() / np.log2(bins)
        return 1 - entropy


class EXAG(AcousticIndex):
    name = 'EXAG'

    def run(self, array):
        power = np.clip(array, 1e-3, 1e+3)**2
        ref = 10*np.log10(power)
        tail = interpercentile_mean_decibels(power, ref, [(90, 100)])
        _, mod_deciles, perc_ranges = decile_mod(ref)
        core = interpercentile_mean_decibels(power, ref, perc_ranges)
        return tail-core
