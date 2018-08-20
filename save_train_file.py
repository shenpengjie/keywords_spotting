import os
import random

import numpy as np
import textgrid as tg
from keras.utils import normalize

import numpy_operation

from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
datalist=[]
lablelist=[]
audio=[]
lab=[]
audio_t=[]
lab_t=[]
flag=0
def handle_data(filename,path,lablename,lablepath):
    (rate,sig)=wav.read(path+'/'+filename)
    mfccfeet=mfcc(sig,rate,winlen=0.02,winfunc=np.hamming)
    delta1=delta(mfccfeet,1)
    delta2=delta(mfccfeet,2)
    feet=np.concatenate((mfccfeet,delta1,delta2),axis=1)
    feet = normalize(feet)
    datalist.append(numpy_operation.get_martix(feet,30,10))
    _lable=read_textgrid(lablepath+'/'+lablename, len(feet))
    _lable=to_one_hot(_lable)
    lablelist.append(_lable)
def to_one_hot(labels, dimension=5):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
def str2int(_str):
    if(_str == ''):
        _str = 0
    else:
        _str = int(_str)
    return _str

def read_textgrid(filename,length):
    wav_textgrid = tg.TextGrid()
    wav_textgrid.read(filename)
    wav_tier = wav_textgrid.getFirst('neidatongxue')
    results = []

    j = 0
    for i in range(length):
        if i * 0.01 >= wav_tier[j].minTime and i*0.01 <=wav_tier[j].maxTime:
            results.append(str2int(wav_tier[j].mark))
        else:
            if(j < len(wav_tier) - 1):
                j = j + 1
                results.append(str2int(wav_tier[j].mark))
            else:
                results.append(0)
    return results
if __name__=="__main__":
    wav_path = "C:/Users/spj/Desktop/train";
    lable_path = "C:/Users/spj/Desktop/train_label"
    lablefilelist = os.listdir(lable_path)
    filelist = os.listdir(wav_path)
    for i in filelist:
        audio.append(i)
    for i in lablefilelist:
        lab.append(i)
    for i in range(len(audio)):
        handle_data(audio[i], wav_path, lab[i], lable_path)
        print(i)
    x = np.concatenate((datalist[0], datalist[1]), 0)
    y = np.concatenate((lablelist[0], lablelist[1]), 0)
    for i in range(2, len(datalist)):
        x = np.concatenate((x, datalist[i]), 0)
        y = np.concatenate((y, lablelist[i]), 0)
        flag=i
    wav_path_t = "C:/Users/spj/Desktop/out";
    lable_path_t = "C:/Users/spj/Desktop/text grid"
    lablefilelist = os.listdir(lable_path)
    filelist = os.listdir(wav_path)
    for i in filelist:
        audio_t.append(i)
    for i in lablefilelist:
        lab_t.append(i)
    for i in range(len(audio_t)):
        handle_data(audio_t[i], wav_path, lab_t[i], lable_path)
        print(i)
    for i in range(flag+1, len(datalist)):
        x = np.concatenate((x, datalist[i]), 0)
        y = np.concatenate((y, lablelist[i]), 0)
        print(flag)
    u = list(range(x.shape[0]))
    random.shuffle(u)
    print(int(x.shape[0] * .10))

    X_t = x[u[:int(x.shape[0] * .10)], :]
    Y_t = y[u[:int(x.shape[0] * .10)], :]

    X = x[u[int(x.shape[0] * .10): int(x.shape[0]*.80)] , :]
    Y = y[u[int(x.shape[0] * .10): int(x.shape[0]*.80)] , :]

    X_validation = x[u[int(x.shape[0] * .80):], :]
    Y_validation = y[u[int(x.shape[0] * .80):], :]
    np.save('./data/train_wav_data.npy',X)
    np.save('./data/train_lable_data.npy',Y)

    np.save('./data/test_wav_data.npy',X_t)
    np.save('./data/test_lable_data.npy', Y_t)

    np.save('./data/validation_wav_data.npy',X_validation)
    np.save('./data/validation_lable_data.npy',Y_validation)