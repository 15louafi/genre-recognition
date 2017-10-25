import os
import glob
import sys
import numpy as np
import scipy
import scipy.io.wavfile
from python_speech_features import mfcc
from python_speech_features import delta

def mfcc(file):
    sample_rate, X = scipy.io.wavfile.read(file)
    mfcc_feat = mfcc(X,sample_rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_d_mfcc_feat = delta(d_mfcc_feat, 2)