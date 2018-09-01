import os
import numpy as np
from keras.utils import normalize
import scipy.io.wavfile as wavfile
from script.utils import read_textgrid
from script.numpy_operation import get_martix
from python_speech_features import mfcc,fbank
from python_speech_features import delta

def to_one_hot(labels, dimension=5):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def read_wav(filename):
    rate, data = wavfile.read(filename)
    #only use the 1st channel if stereo
    if len(data.shape) > 1:
        data =  data[:,0]
    data = data.astype(np.float32)
    data = data / 32768 #convert PCM int16 to float
    return data, rate
    

def feature_extract(filename, wavpath, tgpath):
    filename=os.path.splitext(filename)[0]
    wav_filename =wavpath+'/'+filename+'.wav'
    tg_filename = tgpath+'/'+filename+'.TextGrid'
    
    y,sr = read_wav(wav_filename)
    _mfccs=fbank(signal=y,samplerate=sr,winfunc=np.hamming,winlen=0.02,nfilt=40)[0]
    print(_mfccs.shape)
    # mfccs = mfcc(signal=y,samplerate=sr,winlen=0.02,winfunc=np.hamming)
    # delta1 = delta(mfccs,1)
    # delta2 = delta(mfccs,2)
    #
    # _mfccs = np.concatenate((mfccs,delta1,delta2),1)
    _mfccs = normalize(_mfccs)
    _mfccs = get_martix(_mfccs,30,10)

    _labels = read_textgrid(tg_filename,len(_mfccs))
    _labels = to_one_hot(_labels)
    return _mfccs,_labels
