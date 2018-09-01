import os
import random

import numpy as np


from script.feature_extractor import feature_extract

wav_datas=None
wav_labels=None
# def handle_data(filename,path,lablename,lablepath):
#     (rate,sig)=wav.read(path+'/'+filename)
#     mfccfeet=mfcc(sig,rate,winlen=0.02,winfunc=np.hamming)
#     delta1=delta(mfccfeet,1)
#     delta2=delta(mfccfeet,2)
#     feet=np.concatenate((mfccfeet,delta1,delta2),axis=1)
#     feet = normalize(feet)
#     datalist.append(numpy_operation.get_martix(feet,30,10))
#     _lable=read_textgrid(lablepath+'/'+lablename, len(feet))
#     _lable=to_one_hot(_lable)
#     lablelist.append(_lable)
if __name__=="__main__":
    wav_path = "C:/Users/spj/Desktop/new_data";#本地音频路径
    lable_path = "C:/Users/spj/Desktop/new_lable"#本地标签路径
    filelist = os.listdir(wav_path)
    for i in filelist:
        print(i)
        data,label=feature_extract(i,wav_path,lable_path)


        if wav_datas is None:
            wav_datas = data
            wav_labels = label
        else:
            wav_datas = np.concatenate((wav_datas, data), 0)
            wav_labels = np.concatenate((wav_labels, label), 0)

    x=wav_datas
    y=wav_labels
    u = list(range(x.shape[0]))
    random.shuffle(u)


    X_t = x[u[:int(x.shape[0] * .10)], :]
    Y_t = y[u[:int(x.shape[0] * .10)], :]

    X = x[u[int(x.shape[0] * .10): int(x.shape[0]*.90)] , :]
    Y = y[u[int(x.shape[0] * .10): int(x.shape[0]*.90)] , :]

    X_validation = x[u[int(x.shape[0] * .90):], :]
    Y_validation = y[u[int(x.shape[0] * .90):], :]
    #保存数据路径
    np.save('./data/train_wav_self_fb.npy',X)
    np.save('./data/train_label_self_fb.npy',Y)

    np.save('./data/test_wav_self_fb.npy',X_t)
    np.save('./data/test_label_self_fb.npy', Y_t)

    np.save('./data/validation_wav_self_fb.npy',X_validation)
    np.save('./data/validation_label_self_fb.npy',Y_validation)