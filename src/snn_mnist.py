''' Self Normalizing Neural Network (SNN)
    for MNIST dataset. Using SELU activations with 
    AlphaDropout. 1 input layer, 2 hidden layers with 
    512 hidden units each, and 1 output layer.
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Activation, Dense
from keras.layers.noise import AlphaDropout
from keras.utils import to_categorical, plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()
X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0], -1).astype('float32') / 255
X_TEST = X_TEST.reshape(X_TEST.shape[0], -1).astype('float32') / 255

BATCH = 128
EPOCHS = 20
N_CLASSES = len(np.unique(Y_TRAIN))

Y_TRAIN = to_categorical(Y_TRAIN, N_CLASSES)
Y_TEST = to_categorical(Y_TEST, N_CLASSES)

print('X_TRAIN shape --> {0}'.format(X_TRAIN.shape))
print('Y_TRAIN shape --> {0}'.format(Y_TRAIN.shape))
print('X_TEST shape --> {0}'.format(X_TEST.shape))
print('Y_TEST shape --> {0}'.format(Y_TEST.shape))

IMG_SHAPE = X_TRAIN[0].shape

MODEL = Sequential()
MODEL.add(InputLayer(input_shape=IMG_SHAPE))
MODEL.add(Dense(512))
MODEL.add(Activation('selu'))
MODEL.add(AlphaDropout(0.2))
MODEL.add(Dense(512))
MODEL.add(Activation('selu'))
MODEL.add(AlphaDropout(0.2))
MODEL.add(Dense(N_CLASSES))
MODEL.add(Activation('softmax'))

MODEL.summary()

plot_model(MODEL, to_file='../bin/snn_mnist_2_hidden.png')

with open('../bin/snn_mnist_2_hidden.yaml', 'w') as f:
    f.write(MODEL.to_yaml())

MODEL.compile(loss=categorical_crossentropy,
              optimizer=RMSprop(),
              metrics=['accuracy'])

HIST_MODEL = MODEL.fit(X_TRAIN, Y_TRAIN,
                       batch_size=BATCH,
                       epochs=EPOCHS,
                       verbose=True,
                       validation_data=(X_TEST, Y_TEST))

MODEL.save('../bin/snn_mnist_2_hidden.h5')

LOSS, ACCURACY = MODEL.evaluate(X_TEST, Y_TEST, verbose=False)
print('Test Loss {0}'.format(LOSS))
print('Test Accuracy {0}'.format(ACCURACY))

plt.plot(HIST_MODEL.history['acc'])
plt.plot(HIST_MODEL.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
FIG = plt.gcf()
FIG.savefig('../bin/snn_mnist_2_hidden_accuracy.png')
plt.show()

plt.plot(HIST_MODEL.history['loss'])
plt.plot(HIST_MODEL.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
FIG = plt.gcf()
FIG.savefig('../bin/snn_mnist_2_hidden_loss.png')
plt.show()
