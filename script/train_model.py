
import numpy as np
import keras.callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation,Dropout
import matplotlib.pyplot as plt
import tensorflow as tf



early_stopping=keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=0,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0
)




def train_(x,y,test_x,test_y,val_x,val_y):

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(39 * 41,)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  # Fit the model

    history = model.fit(x, y, epochs=50, batch_size=512, validation_data=(val_x, val_y),callbacks=[early_stopping])
    print(model.evaluate(test_x, test_y, batch_size=512))

    model.save('self_model.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']


    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Traing and validtion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result.jpg')
    plt.show()





if __name__=="__main__":
#本地把音频和标签分别保存成numpy数组
    x=np.load('data/train_wav_self.npy')
    y=np.load('data/train_label_self.npy')
    test_x=np.load('data/test_wav_self.npy')
    test_y=np.load('data/test_label_self.npy')
    validation_x=np.load('data/validation_wav_self.npy')
    validation_y=np.load('data/validation_label_self.npy')
    train_(x, y, test_x, test_y, validation_x, validation_y)
