# from wxpy import *
# bot=Bot()
# @bot.register(Group ,RECORDING)
# def get_recording(msg):
#     msg.get_file(''+msg.file_name)
#     print(msg.file_name+'已下载')
# embed()
import os
import random

import librosa
import numpy as np
from keras.utils import normalize

from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav

from script.feature_extractor import to_one_hot
from script.numpy_operation import get_martix

def pre(path,label):
    X = []
    Y = []
    filelist = os.listdir(path)
    j = 0
    for i in filelist:

        i=os.path.join(path,i)
        a = []
        num,feature=loadwac(i)
        if(len(X)==0):
            X=feature
            for i in range(num):
                a.append(label)
            Y=a
        else:
            X=np.concatenate((X,feature),axis=0)
            for i in range(num):
                a.append(label)
            Y=np.concatenate((Y,a),axis=0)
        j=j+1
        print(j)

    Y = to_one_hot(Y,3)
    return X,Y


def loadwac(file_path):
    rate,data=wav.read(file_path)
    mfccfeet=mfcc(data,rate,winlen=0.02,winfunc=np.hamming)
    delta1 = delta(mfccfeet, 1)
    delta2 = delta(mfccfeet, 2)
    _mfccs = np.concatenate((mfccfeet, delta1, delta2), 1)
    _mfccs = normalize(_mfccs)
    _mfccs = get_martix(_mfccs, 30, 10)
    frame=_mfccs.shape[0]
    return frame,_mfccs



if __name__=="__main__":
    x,y=pre('C:/Users/spj/Desktop/google_data/cat',2)
    print('1')
    x1,y1=pre('C:/Users/spj/Desktop/google_data/six',1)
    print('2')
    x2,y2=pre('C:/Users/spj/Desktop/google_data/noise',0)
    print('3')
    x3,y3=pre('C:/Users/spj/Desktop/google_data/filter',0)
    x=np.concatenate((x,x1,x2,x3),axis=0)
    y=np.concatenate((y,y1,y2,y3),axis=0)

    u = list(range(x.shape[0]))
    random.shuffle(u)

    X_t = x[u[:int(x.shape[0] * .10)], :]
    Y_t = y[u[:int(x.shape[0] * .10)], :]

    X = x[u[int(x.shape[0] * .10): int(x.shape[0] * .90)], :]
    Y = y[u[int(x.shape[0] * .10): int(x.shape[0] * .90)], :]

    X_validation = x[u[int(x.shape[0] * .90):], :]
    Y_validation = y[u[int(x.shape[0] * .90):], :]
    np.save('./data/train_wav_google.npy', X)
    np.save('./data/train_label_google.npy', Y)

    np.save('./data/test_wav_data.npy', X_t)
    np.save('./data/test_label_google.npy', Y_t)

    np.save('./data/validation_wav_google.npy', X_validation)
    np.save('./data/validation_label_google.npy', Y_validation)