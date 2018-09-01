import string

from keras.utils import normalize
from python_speech_features import mfcc,fbank
from python_speech_features import delta
import script.numpy_operation
import scipy.io.wavfile as wav
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# def dtw_distance(mat, w=5):
#     # function [d,D,T] = dtw_distance(mat, w)
#     inf = 10000000
#     D = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1)) + float(inf);
#     D[0, 0] = 0;
#     # T = zeros(size(mat)+1);
#     ns, nt = mat.shape;
#     w = max(w, abs(ns - nt));
#     for i in range(ns):
#         for j in range(max(i - w, 0), min(i + w, nt-1) + 1):
#             oost = -mat[i, j];
#             D[i + 1, j + 1] = oost + min([D[i, j + 1], D[i + 1, j], D[i, j]]);
#     d = -D[ns, nt] / max(nt, ns);
#     return d
(rate,sig)=wav.read('test2.wav')
# feet=fbank(sig,rate,winlen=0.02,winfunc=np.hamming,nfilt=40)[0]
mfccfeet=mfcc(sig,rate,winlen=0.02,winfunc=np.hamming)
delta1=delta(mfccfeet,1)
delta2=delta(mfccfeet,2)
feet=np.concatenate((mfccfeet,delta1,delta2),axis=1)
feet=normalize(feet)

test=script.numpy_operation.get_martix(feet,30,10)
test=test.reshape((test.shape[0],39,41,1))

wsmooth=30
list=[]
model=load_model('new_model\cov_model_min.h5')
predictions=model.predict(test)

pospredictions=np.zeros(predictions.shape)
for i in range(predictions.shape[1]):
    for j in range(predictions.shape[0]):
        hsmooth=max(0,j-wsmooth)
        for k in range(hsmooth,j+1):
            pospredictions[j][i] +=predictions[k][i]
        pospredictions[j][i]=pospredictions[j][i]/(j-hsmooth+1)
confidence=np.zeros(pospredictions.shape[0])
wmax = 50
for j in range(pospredictions.shape[0]):
    confidence[j] = 1
    hmax = max(0, j - wmax)
    for i in range(1, 5):
        confidence[j] *= max(pospredictions[hmax:j + 1, i])
    confidence[j] = pow(confidence[j], 1 / 4)

for i in range(predictions.shape[0]):
    list.append(np.argmax(pospredictions[i]))

p=range(1,pospredictions.shape[0]+1)
# l1=[i for i in range(len(list)) if list[i] == 1]
# l2=[i for i in range(len(list)) if list[i] == 2]
# l3=[i for i in range(len(list)) if list[i] == 3]
# l4=[i for i in range(len(list)) if list[i] == 4]
# print(l1,l2,l3,l4)
# max1=0
# index1=0
# for i in l1:
#     if(max1<confidence[i]):
#         max1=confidence[i]
#         index1=i
# max2=0
# index2=0
# for i in l2:
#     if(max2<confidence[i]):
#         max2=confidence[i]
#         index2=i
#
# max3=0
# index3=0
# for i in l3:
#     if(max3<confidence[i]):
#         max3=confidence[i]
#         index3=i
#
# max4=0
# index4=0
# for i in l4:
#     if(max4<confidence[i]):
#         max4=confidence[i]
#         index4=i
# print(max1,max2,max3,max4)
# if(max1>=0.85and max2>=0.85 and max3>=0.85 and max4>=0.85 and index1<index2 and index2<index3 and index3<index4):
#     print('true')
# else:
#     print('false')
# print(max4,max3,max2,max1)
# print(confidence[l1],confidence[l2],confidence[l3],confidence[l4])
plt.plot(p,list, 'b')
plt.xlabel('frame')
plt.ylabel('confidence')
plt.show()
print(max(confidence))

