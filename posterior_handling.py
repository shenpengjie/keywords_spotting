from keras.utils import normalize
from python_speech_features import mfcc
from python_speech_features import delta
import numpy_operation
import scipy.io.wavfile as wav
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
(rate,sig)=wav.read('test1.wav')
mfccfeet=mfcc(sig,rate,winlen=0.02,winfunc=np.hamming)
delta1=delta(mfccfeet,1)
delta2=delta(mfccfeet,2)
feet=np.concatenate((mfccfeet,delta1,delta2),axis=1)
feet=normalize(feet)
test=numpy_operation.get_martix(feet,30,10)
wsmooth=30
list=[]
model=load_model('model.h5')
predictions=model.predict(test)
print(predictions.shape)
for i in range(predictions.shape[0] ):
    print(np.argmax(predictions[i]))
    # list.append(np.argmax(predictions[i]))
pospredictions=np.zeros(predictions.shape)
for i in range(predictions.shape[1]):
    for j in range(predictions.shape[0]):
        hsmooth=max(0,j-wsmooth)
        for k in range(hsmooth,j):
            pospredictions[j][i]+=predictions[k][i]
        pospredictions[j][i]=pospredictions[j][i]/(j-hsmooth+1)
wmax=100
confidence=np.zeros(pospredictions.shape[0])
for j in range(pospredictions.shape[0]):
    confidence[j] = 1
    hmax = max(0, j - wmax)
    for i in range(0,4):
        confidence[j]*=max(pospredictions[hmax:j+1,i])#[k][i] for k in range(hmax,j+1))
    confidence[j]=pow(confidence[i],1/4)
for i in range(predictions.shape[0]):
    list.append(np.argmax(pospredictions[i]))
p=range(1,pospredictions.shape[0]+1)
plt.plot(p,list, 'b')
plt.xlabel('frame')
plt.ylabel('confidence')
plt.show()