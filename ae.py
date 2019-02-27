from matplotlib import pyplot as plt 
from scipy.misc import toimage
import keras
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
import os
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

UNLABELED = './data/stl10_binary/unlabeled_X.bin'
TEST_PATH = './data/stl10_binary/test_X.bin'

def load_stl10_unlabeled_data():
    with open(UNLABELED, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)
        unlabeled = np.reshape(images, (-1, 3, 96, 96))
        unlabeled = np.transpose(unlabeled, (0, 3, 2, 1))

    return unlabeled

def load_stl10_test_data():
    with open(TEST_PATH, 'rb') as f:
        testx = np.fromfile(f, dtype=np.uint8)
        x_test = np.reshape(testx, (-1, 3, 96, 96))
        x_test = np.transpose(x_test, (0, 3, 2, 1))

    return x_test

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


unlabeled = load_stl10_unlabeled_data()
x_test = load_stl10_test_data()

unlabeled = unlabeled.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(unlabeled.shape)
print(x_test.shape)

#batch_size = 32
#epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'autoencoder.h5'

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', input_shape=(96, 96, 3)))
model.add(Activation('relu'))

model.add(Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(9216))
model.add(Activation('relu'))

model.add(Reshape((24,24,16)))

model.add(Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.summary()



model.compile(loss='binary_crossentropy',
	      optimizer='adam')

#checkpoint
filepath='./saved_models/3/autoencoder_weights.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=10)
callbacks_list = [checkpoint]

batch_size = 32
epochs = 50 

history = model.fit(unlabeled, unlabeled,
          batch_size=batch_size,
          epochs=epochs,
	  verbose=1,
  	  callbacks=callbacks_list,
          validation_data=(x_test,  x_test),
          shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('result.png')

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)

#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)



