from matplotlib import pyplot
from scipy.misc import toimage
import keras
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.utils import to_categorical

TRAIN_PATH = './data/stl10_binary/train_X.bin'
LABEL_PATH = './data/stl10_binary/train_y.bin'
TEST_PATH = './data/stl10_binary/test_X.bin'
TEST_LABEL = './data/stl10_binary/test_y.bin'


def load_stf10_data():
    with open(TRAIN_PATH, 'rb') as f:
        trainx = np.fromfile(f, dtype=np.uint8)
        x_train = np.reshape(trainx, (-1, 3, 96, 96))
        x_train = np.transpose(x_train, (0, 3, 2, 1))

    with open(LABEL_PATH, 'rb') as f:
        y_train = np.fromfile(f, dtype=np.uint8)

    with open(TEST_PATH, 'rb') as f:
        testx = np.fromfile(f, dtype=np.uint8)
        x_test = np.reshape(testx, (-1, 3, 96, 96))
        x_test = np.transpose(x_test, (0, 3, 2, 1))

    with open(TEST_LABEL, 'rb') as f:
        y_test = np.fromfile(f, dtype=np.uint8)

    return (x_train, y_train), (x_test, y_test)


#def show_imgs(X):
#    pyplot.figure(1)
#    k = 0
#    for i in range(0, 4):
#        for j in range(0, 4):
#            pyplot.subplot2grid((4, 4), (i, j))
#            pyplot.imshow(toimage(X[k]))
#            k = k+1
#    # show the plot
#    pyplot.show()


(x_train, y_train), (x_test, y_test) = load_stf10_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print((x_train.shape))
print((y_train.shape))
print((x_test).shape)
print((y_test).shape)

num_classes = 10
y_train = np.reshape(y_train, (5000, 1))
y_test = np.reshape(y_test, (8000, 1))


for i in range(len(y_train)):
    y_train[i] = int(y_train[i]) - 1


for i in range(len(y_test)):
    y_test[i] = int(y_test[i]) - 1


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 32
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

model = Sequential()

model.add(Conv2D(16, (3, 3), strides=2, input_shape=(96, 96, 3)))
model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3), strides=2))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.fit(x_train / 255.0, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test / 255.0, y_test),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

