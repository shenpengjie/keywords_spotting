import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(39,41,1)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def train(x,y,test_x,test_y,val_x,val_y):
    model=get_model()
    history=model.fit(x, y, epochs=20, batch_size=512,verbose=1, validation_data=(val_x, val_y))
    print(model.evaluate(test_x, test_y, batch_size=512))
    model.save('model\cov_model.h5')
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Traing and validtion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result/result.jpg')
    plt.show()



if __name__=="__main__":

    x=np.load('data/train_wav_google.npy')
    y=np.load('data/train_label_google.npy')
    test_x=np.load('data/test_wav_google.npy')
    test_y=np.load('data/test_label_google.npy')
    validation_x=np.load('data/validation_wav_google.npy')
    validation_y=np.load('data/validation_label_google.npy')

    x=x.reshape((x.shape[0],39,41,1))
    test_x = test_x.reshape((test_x.shape[0], 39, 41,1))
    validation_x = validation_x.reshape((validation_x.shape[0], 39, 41,1))



    train(x, y, test_x, test_y, validation_x, validation_y)