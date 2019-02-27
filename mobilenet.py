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
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

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


#y_train = to_categorical(y_train, num_classes)
#y_test = to_categorical(y_test, num_classes)


model = MobileNet(weights='imagenet', include_top=False)

print(y_train.shape)

new_x_test = []
for img in x_test:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = model.predict(x)
    
    x = x.flatten()
    
    new_x_test.append(x)

codigos = []
for img in x_train:
    # print(img.shape)
    x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    
    codigos.append(features)

codigos = np.asarray(codigos)    
print(codigos.shape)

print('Treinando SVM...')
clf = svm.SVC(kernel='linear', C=1)
clf.fit(codigos, y_train)

print('Calculando score...')
score = clf.score(new_x_test, y_test)
print('\nscore: ' + str(score))
