from matplotlib import pyplot as plt 
from scipy.misc import toimage
import keras
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
import os
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import time
import sys
from contextlib import redirect_stdout
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

UNLABELED = './data/stl10_binary/unlabeled_X.bin'
TEST_PATH = './data/stl10_binary/test_X.bin'

TEST_ID = int(time.time())
SAVE_PATH = sys.argv[1]
#rateio = float(sys.argv[2])
print(SAVE_PATH)

#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = str(TEST_ID) + '.h5'

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

def custom_loss(x1,x2,x2_hat,x1_hat):

    
    def loss(y_true, y_pred):
        
       
        return K.mean(math_ops.square(y_pred - y_true), axis=-1) + 0.5*K.mean(math_ops.square(x1_hat - x1), axis=-1) + 0.5*K.mean(math_ops.square(x2_hat - x2), axis=-1)
       
    return loss

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

#
input_img = Input(shape=(96,96,3))

# encoder
x1 = Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(input_img)

print('x1: ')
print(x1.shape)

x2 = Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(x1)

print('x2: ')
print(x2.shape)

x2f = Flatten()(x2)

# code
z = Dense(1024, activation='relu')(x2f)

# decoder
x2f_hat = Dense(9216, activation='relu')(z)

x2_hat = Reshape((24,24,16))(x2f_hat)

print('x2_hat: ')
print(x2_hat.shape)

x1_hat = Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(x2_hat)

print('x1_hat: ')
print(x1_hat.shape)


reconst = Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(x1_hat)

autoencoder = Model(input_img, reconst)
autoencoder.summary()
#



summary_save_name = './' + str(SAVE_PATH) + str(TEST_ID) + '.txt'
with open(summary_save_name, 'w') as f:
    with redirect_stdout(f):
        autoencoder.summary()

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.00001) 

#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.7, epsilon=None, decay=0.0)

autoencoder.compile(loss=custom_loss(x1,x2,x2_hat,x1_hat),
                    optimizer=opt)

#checkpoint
filepath= './' +  str(SAVE_PATH) + str(TEST_ID) + '.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=5)
callbacks_list = [checkpoint]

batch_size = 32
epochs = 100

history = autoencoder.fit(unlabeled, unlabeled,
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
save_fig_path = './' + str(SAVE_PATH) +  str(TEST_ID) + '.png'
plt.savefig(save_fig_path)

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)

#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)



