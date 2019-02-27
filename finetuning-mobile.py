from matplotlib import pyplot
from scipy.misc import toimage
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
import os
from keras.utils import to_categorical
from keras.applications.mobilenet import MobileNet
from keras import layers
from keras import optimizers
from keras import models
import numpy as np
import skimage
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



(x_train, y_train), (x_test, y_test) = load_stf10_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#y_train = to_categorical(y_train, num_classes=10, dtype='float32')
#y_teste = to_categorical(y_teste, num_classes=10, dtype='float32')


x_train= [skimage.transform.resize(image, (224,224,3)) for image in x_train]
x_test = [skimage.transform.resize(image, (224,224,3)) for image in x_test]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
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


model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print(y_train.shape)

for layer in model.layers[:-3]:
    layer.trainable = False

for layer in  model.layers:
    print(layer, layer.trainable)



new_model = models.Sequential()

new_model.add(model)

new_model.add(layers.Flatten())
new_model.add(layers.Dense(512, activation='relu'))
new_model.add(layers.Dense(10, activation='softmax'))

new_model.summary()

batch_size = 32
epochs = 100

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


new_model.fit(x_train / 255.0, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test / 255.0, y_test),
          shuffle=True)

save_dir = './finetuning/'
model_name = 'mobile.hdf5'
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
