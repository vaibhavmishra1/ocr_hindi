from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
from keras.models import Sequential
from keras.constraints import maxnorm
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import numpy as np
from mnist import MNIST
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json


mnistdata = MNIST('data')
X_training,y_training = mnistdata.load('EMNIST_data/emnist-letters-train-images-idx3-ubyte','EMNIST_data/emnist-letters-train-labels-idx1-ubyte')
X_testing,Y_testing = mnistdata.load('EMNIST_data/emnist-letters-test-images-idx3-ubyte','EMNIST_data/emnist-letters-test-labels-idx1-ubyte')

print("vaibhav-:dataset loaded")
X_training = np.array(X_training) / 255.0
y_training = np.array(y_training)
X_testing = np.array(X_testing) / 255.0
Y_testing = np.array(Y_testing)
X_training = X_training.reshape(X_training.shape[0], 28, 28)
X_testing = X_testing.reshape(X_testing.shape[0], 28, 28)

for t in range(124800):
    X_training[t]=np.transpose(X_training[t])
 
for t in range(20800):
    X_testing[t]=np.transpose(X_testing[t])

X_training = X_training.reshape(X_training.shape[0], 784,1)
X_testing = X_testing.reshape(X_testing.shape[0], 784,1)

def reshaping(ipar):
    opar = []
    for image in ipar:
        opar.append(image.reshape(-1))
    return np.asarray(opar)



train_images = X_training.astype('float32')
test_images = X_testing.astype('float32')

train_images = reshaping(train_images)
test_images = reshaping(test_images)


train_labels = np_utils.to_categorical(y_training, 62)
test_labels = np_utils.to_categorical(Y_testing, 62)

K.set_learning_phase(1)

model = Sequential()
def create_model():
  model.add(Reshape((28,28,1), input_shape=(784,)))
  model.add(Convolution2D(32, (5,5), input_shape=(28,28,1),
                               activation='relu',padding='same',
                              kernel_constraint=maxnorm(3)))

  model.add(Convolution2D(32, (5,5),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.5))
  model.add(Dense(62, activation='softmax'))

  opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  print(model.summary())

create_model()
history = model.fit(train_images,train_labels,validation_data=(test_images, test_labels), batch_size=128, epochs=20)
scores = model.evaluate(test_images,test_labels, verbose = 0)
print("Accuracy: %.2f%%"%(scores[1]*100))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()


model_json = model.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_new.h5")



