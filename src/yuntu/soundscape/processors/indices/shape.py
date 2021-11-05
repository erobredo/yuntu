import numpy as np
import pywt
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy import stats
from yuntu.soundscape.processors.indices.base import AcousticIndex


def antialiased_resize(im, new_size):
    r = float(new_size[0]) / im.shape[0]
    im = gaussian_filter(im, [0.25/r, 0.25/r])
    im = resize(im, new_size)
    return im

def pack_wavelet_coeffs(w, level=4):
    pw = np.zeros(2**((level-1)*2), dtype=np.float32)

    pw[0:len(np.ravel(w[0]))] = np.ravel(w[0])
    ni = len(np.ravel(w[0]))

    for i in range(1,level):
        for j in range(0,3):
            pw[ni:ni+len(np.ravel(w[i][j]))] = np.ravel(w[i][j])
            ni += len(np.ravel(w[i][j]))
    return pw

def get_sparse_haar_coefficients(im, ncoeffs = 40, level=4):
    w = pywt.wavedec2(im, 'Haar')
    pw = pack_wavelet_coeffs(w, level=level)
    pw = pw / np.sqrt(np.sum(pw**2))

    return pw

def get_fingerprint(array, ncoeffs=40, level=2, latus=128):
    norm_array = stats.zscore(array)
    im = antialiased_resize(array, [latus, latus])
    fingerprint = get_sparse_haar_coefficients(im, ncoeffs=ncoeffs, level=level)
    return fingerprint

class WAVELET(AcousticIndex):
    name = 'WAVELET_FINGERPRINT'

    def __init__(self,*args, ncoeffs=40, level=2, latus=128, **kwargs):
        self.ncoeffs = ncoeffs
        self.latus = latus
        self.level = level
        self.ncomponents = 2**((level-1)*2)
        super().__init__(*args,**kwargs)

    def run(self, array, **kwargs):
        return get_fingerprint(array, **kwargs)
