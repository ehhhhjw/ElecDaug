import pickle
import numpy as np
import keras.models as models
from keras.layers.core import  Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, ZeroPadding2D, Reshape
from keras.utils import to_categorical
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, \
    VirtualAdversarialMethod, MomentumIterativeMethod, Noise, LBFGS, BasicIterativeMethod
import keras.backend as K
import matplotlib.pyplot as plt
import time
import os
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Xd = pickle.load(open('/home/ise/dl/imageandwave/adversarial/RML2016.10a_dict.pkl', 'rb'), encoding='bytes')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
snr = 0

for mod in mods:
    elem = (mod, snr)
    X.append(Xd[elem])
    for i in range(Xd[elem].shape[0]):
        lbl.append(elem)
print(mods)
X = np.vstack(X)  #X.shape=(11000, 2, 128)
np.random.seed(2017)
n_examples = X.shape[0]
n_train = 9000
train_idx = np.random.choice(range(n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]
Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)


dr = 0.5
classes = mods
print(classes)
in_shp = list(X_train.shape[1:])
print("in_shp:", in_shp)
model = models.Sequential()  # Neural network is a set of sequential layers
model.add(Reshape(in_shp+[1], input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))  # Add 2 columns of zeros to each side
model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2))) # Add 2 columns of zeros to each side
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2"))
softmax = model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('/home/ise/dl/imageandwave/adversarial/CNN_%ddB.hdf5' % snr, monitor="val_acc", verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor="val_acc", patience=30)
#
model.fit(X_train, Y_train, batch_size=1024, epochs=500, verbose=1, validation_data=(X_test, Y_test),
            callbacks=[checkpoint, earlystopping])
model.save_weights('original_%ddB.hdf5' % snr)
br = model.evaluate(X_test, Y_test, verbose=0)[1]
print("Before Attack: ", br)