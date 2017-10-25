import os
import glob
import sys
import numpy as np
import scipy
import scipy.io.wavfile
from python_speech_features import mfcc
from python_speech_features import delta
genre_list = []

def extract_mfcc(file):
    """
        MFCC coefficients computation
    """ 
    sample_rate, X = scipy.io.wavfile.read(file)
    mfcc_feat = mfcc(X,sample_rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_d_mfcc_feat = delta(d_mfcc_feat, 2)
    feature=np.hstack((mfcc_feat,d_mfcc_feat,d_d_mfcc_feat))
    write_mfcc(file,feature)
    print(feature)

def write_mfcc(file,mfcc):
    """
        Saves the MFCC features into a file
    """
    filename, extension = os.path.splitext(file)
    datafile = filename + ".mfcc"
    np.save(datafile, mfcc)
    

def read_mfcc():
    """
        Reads all the MFCC files and outputs them into a single matrix
    """
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for file in glob.glob(os.path.join("files", genre, "*.mfcc.npy")):
            X = []
            mfcc = np.load(file)
            X.append(mfcc.flatten()) #One dimensional feature vector, so we have to flatten (or apply PCA)
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    for subdir, dirs, files in os.walk("files"):
        inter = list(set(dirs).intersection( set(genre_list) ))
        break
    print("Genres found : ", inter)
    print("Feature creation, please wait...")   
    for subdir, dirs, files in os.walk("files"):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                genre = subdir[subdir.rfind('/',0)+1:]
                if genre in inter:
                extract_mfcc(path)
    print("Feature generation done !")
