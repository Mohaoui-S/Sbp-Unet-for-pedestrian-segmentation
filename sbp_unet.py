import os

from zipfile import ZipFile
file_name = 'data_ochuman.zip'
with ZipFile(file_name, 'r') as zip1:
  zip1.extractall()
  del zip1

os.remove("data_ochuman.zip")

git clone https://github.com/liruilong940607/OCHumanApi
os.chdir("OCHumanApi")
make install
os.chdir("../")
os.listdir()

# Commented out IPython magic to ensure Python compatibility.
from ochumanApi.ochuman import OCHuman
import cv2, os
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 15)

import ochumanApi.vis as vistool
from ochumanApi.ochuman import Poly2Mask

import tensorflow as tf
from keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split

from google.colab.patches import cv2_imshow

from ochumanApi.ochuman import OCHuman
ochuman = OCHuman(AnnoFile='ochuman.json', Filter='segm')
image_ids = ochuman.getImgIds()
#print ('Total images: %d'%len(image_ids))

IMG_HEIGHT = 512
IMG_WIDTH = 512
epochs = 500
batch_size = 8
ImgDir = "custom_dataset_human_black_background/"

features = os.listdir(f"{ImgDir}features/")
labels = os.listdir(f"{ImgDir}labels/")

#print(len(features), len(labels))

X = features
y = labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.15, random_state=1)

print(len(X_train), len(X_val), len(X_test))

def keras_generator_train_val_test(batch_size, choice="train"):

    if choice == "train":
        X = X_train
        y = y_train
    elif choice == "val":
        X = X_val
        y = y_val
    elif choice == "test":
        X = X_test
        y = y_test
    else:
        print("Invalid Option")
        return False

    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            x_rand = random.choice(X)
            y_rand = x_rand[:-5]+"y.jpg"

            x_path = f"{ImgDir}features/{x_rand}"
            y_path = f"{ImgDir}labels/{y_rand}"

            x = cv2.imread(x_path)
            y = cv2.imread(y_path)

            x = x / 255.
            y = y / 255.

            x_batch.append(x)
            y_batch.append(y)


        x_batch = np.array(x_batch)
        # y_batch = np.array(y_batch)

        y_batch = {'seg': np.array(y_batch),
                #    'cls': np.array(classification_list)
                }

        yield x_batch, y_batch

for x, y in keras_generator_train_val_test(2, choice="train"):
    break

#print(x.shape, y['seg'].shape)

def fire_module(x, fire_id, squeeze=16, expand=64):
    f_name = "fire{0}/{1}"
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(squeeze, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, "squeeze1x1"))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    left = Conv2D(expand, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, "expand1x1"))(x)
    right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=f_name.format(fire_id, "expand3x3"))(x)
    x = concatenate([left, right], axis=channel_axis, name=f_name.format(fire_id, "concat"))
    return x


def SBpUNet(deconv_ksize=3, dropout=0.5, activation='sigmoid'):
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation
    :param inputs: input layer.
    :param num_classes: number of classes.
    :param deconv_ksize: (width and height) or integer of the 2D deconvolution window.
    :param dropout: dropout rate
    :param activation: type of activation at the top layer.
    :returns: SBp-UNet model
    """
    in1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3 ))

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x01 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1')(in1)
    x02 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1', padding='same')(x01)

    x03 = fire_module(x02, fire_id=2, squeeze=16, expand=64)
    x04 = fire_module(x03, fire_id=3, squeeze=16, expand=64)
    x05 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3', padding="same")(x04)

    x06 = fire_module(x05, fire_id=4, squeeze=32, expand=128)
    x07 = fire_module(x06, fire_id=5, squeeze=32, expand=128)
    x08 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding="same")(x07)

    x09 = fire_module(x08, fire_id=6, squeeze=48, expand=192)
    x10 = fire_module(x09, fire_id=7, squeeze=48, expand=192)
    x11 = fire_module(x10, fire_id=8, squeeze=64, expand=256)
    x12 = fire_module(x11, fire_id=9, squeeze=64, expand=256)

    if dropout != 0.0:
        x12 = Dropout(dropout)(x12)

    up1 = concatenate([Conv2DTranspose(192, deconv_ksize, strides=(1, 1), padding='same')(x12),x10,], axis=channel_axis)
    up1 = fire_module(up1, fire_id=10, squeeze=48, expand=192)

    c10 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x10)
    c10 = Conv2DTranspose(128, deconv_ksize, strides=(2, 2), padding='same')(c10)
    up2 = concatenate([Conv2DTranspose(128, deconv_ksize, strides=(1, 1), padding='same')(up1)+c10,x08,], axis=channel_axis)
    up2 = fire_module(up2, fire_id=11, squeeze=32, expand=128)

    c8 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x07)
    c8 = Conv2DTranspose(64, deconv_ksize, strides=(2, 2), padding='same')(c8)
    up3 = concatenate([Conv2DTranspose(64, deconv_ksize, strides=(2, 2), padding='same')(up2)+c8,x05,], axis=channel_axis)
    up3 = fire_module(up3, fire_id=12, squeeze=16, expand=64)

    c5 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x04)
    c5 = Conv2DTranspose(32, deconv_ksize, strides=(2, 2), padding='same')(c5)
    up4 = concatenate([Conv2DTranspose(32, deconv_ksize, strides=(2, 2), padding='same')(up3)+c5,x02,], axis=channel_axis)
    up4 = fire_module(up4, fire_id=13, squeeze=16, expand=32)
    up4 = UpSampling2D(size=(2, 2))(up4)

    x = concatenate([up4, x01], axis=channel_axis)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(3, (1, 1), activation=activation, name='seg')(x)

    model = Model(inputs=[in1], outputs=[x])

    losses = {'seg': 'binary_crossentropy'
            }

    metrics = {'seg': ['acc']
                }
    model.compile(optimizer="adam", loss = losses, metrics=metrics)

    return model

import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs={}):

        res_dir = "intermediate_results_black_background"

        try:
            os.makedirs(res_dir)
        except:
            print(f"{res_dir} directory already exist")

        print('Training: epoch {} begins at {}'.format(epoch, datetime.datetime.now().time()))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
          res_dir = "intermediate_results_black_background/"
          print('Training: epoch {} ends at {}'.format(epoch, datetime.datetime.now().time()))

          for x_test, y_test in keras_generator_train_val_test(batch_size, choice="test"):
              break
          p = np.reshape(x_test[0], (1, 512, 512, 3))
          prediction = self.model.predict(p)

          x_img = f"{res_dir}{epoch}_X_input.jpg"
          y_img = f"{res_dir}{epoch}_Y_truth.jpg"
          predicted_img = f"{res_dir}{epoch}_Y_predicted.jpg"

          cv2.imwrite(x_img, x_test[0] * 255.)
          cv2.imwrite(y_img, y_test['seg'][0] * 255.)
          cv2.imwrite(predicted_img, prediction[0] * 255.)

model = SBpUNet()
#model = load_model('OurModel.h5')

model.summary()

model_name = "models/"+"SBp-UNet.h5"

#History
filename='log.csv'
history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

modelcheckpoint = ModelCheckpoint(model_name,
                                  monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True)

#lr_callback = ReduceLROnPlateau(min_lr=0.000001)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

callback_list = [modelcheckpoint, earlyStopping, reduce_lr_loss, MyCustomCallback(), history_logger]

history = model.fit_generator(
    keras_generator_train_val_test(batch_size, choice="train"),
    validation_data = keras_generator_train_val_test(batch_size, choice="val"),
    validation_steps = int( np.floor(len(X_val) / batch_size) ),
    steps_per_epoch=int( np.floor(len(X_train) / batch_size) ),
    epochs=epochs,
    #initial_epoch = 46,
    verbose=1,
    shuffle=True,
    callbacks = callback_list,
)