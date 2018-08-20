
import numpy as np
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import matplotlib.pyplot as plt



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
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  # Fit the model

    history = model.fit(x, y, epochs=17, batch_size=512, validation_data=(val_x, val_y),callbacks=[early_stopping])
    print(model.evaluate(test_x, test_y, batch_size=512))
    model.save('model2.h5')
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

    x=np.load('data/train_wav.npy')
    y=np.load('data/train_label.npy')
    test_x=np.load('data/test_wav.npy')
    test_y=np.load('data/test_label.npy')
    validation_x=np.load('data/val_wav.npy')
    validation_y=np.load('data/val_label.npy')

    train_(x, y, test_x, test_y, validation_x, validation_y)
