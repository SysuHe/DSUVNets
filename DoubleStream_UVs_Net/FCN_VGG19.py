from keras.engine import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

from Net_utils import BilinearUpSampling, conv_act_block

import os
import time
import gdal
import cv2
import numpy as np
import pandas as pd


def load_image_gdal(fname, mode="color", target_size=None):
    if mode == "color":
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
        col = ds.RasterXSize
        row = ds.RasterYSize
        band = ds.RasterCount

        img = np.zeros((row, col, band))
        for i in range(band):
            dt = ds.GetRasterBand(i + 1)
            img[:, :, i] = dt.ReadAsArray(0, 0, col, row)
    else:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    if target_size is not None:
        img = cv2.resize(img, dsize=target_size)

    return img

def load_image_hsr(image_path, is_gray=False, target_size=None):
    if is_gray:
        try:
            img = img_to_array(load_img(image_path, color_mode="grayscale", target_size=target_size))
        except:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if target_size is not None:
                img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=-1)
    else:
        try:
            img = img_to_array(load_img(image_path, target_size=target_size))
        except:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if target_size is not None:
                img = cv2.resize(img, target_size)

    return img

def tradata_generator(base_urls, traimage_dir, tralabel_dir, image_width, image_height, batch_size, shuffle=True):
    while True:
        if shuffle:
            np.random.shuffle(base_urls)
        images = []
        labels = []
        batch_i = 0
        for i in range(len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            # img = load_image_gdal(os.path.join(traimage_dir, base_url + ".tif"), mode="color", target_size=(image_width, image_height))
            # label = load_image_gdal(os.path.join(tralabel_dir, base_url + ".tif"), mode="gray", target_size=(image_width, image_height))
            img = load_image_hsr(os.path.join(traimage_dir, base_url + ".tif"), is_gray=False)
            label = load_image_hsr(os.path.join(tralabel_dir, base_url + ".tif"), is_gray=True)

            label = np.where(label>=2, 0, label)
            label = to_categorical(label, 2)
            images.append(img)
            labels.append(label)
            if batch_i == batch_size:
                train_data = np.array(images)
                train_label = np.array(labels)
                yield(train_data, train_label)
                images = []
                labels = []
                batch_i = 0

def valdata_generator(base_urls, valimage_dir, vallabel_dir, image_width, image_height, batch_size):
    while True:
        images = []
        labels = []
        batch_i = 0
        for i in range(len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            # img = load_image_gdal(os.path.join(valimage_dir, base_url + ".tif"), mode="color", target_size=(image_width, image_height))
            # label = load_image_gdal(os.path.join(vallabel_dir, base_url + ".tif"), mode="gray", target_size=(image_width, image_height))
            img = load_image_hsr(os.path.join(valimage_dir, base_url + ".tif"), is_gray=False)
            label = load_image_hsr(os.path.join(vallabel_dir, base_url + ".tif"), is_gray=True)

            label = np.where(label>=2, 0, label)
            label = to_categorical(label, 2)
            images.append(img)
            labels.append(label)
            if batch_i == batch_size:
                val_data = np.array(images)
                val_label = np.array(labels)
                yield (val_data, val_label)
                images = []
                labels = []
                batch_i = 0

def learning_rate_schedule(_epoch):
    base_learning_rate = 1e-4
    epoch = 100
    lr_base = base_learning_rate
    lr = lr_base * ((1 - float(_epoch) / epoch) ** 0.9)
    return lr

def FCN_VGG19(input_shape,
              n_class,
              weight_decay=1e-4,
              kernel_initializer="he_normal"):

    input_x = Input(shape=input_shape)

    x = conv_act_block(input_x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = conv_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    mark_2 = x

    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    mark_1 = x

    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = conv_act_block(x, 1024, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 1024, weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    x = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling(target_size=(input_shape[0] // 16, input_shape[1] // 16))(x)
    mark_1 = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
                    kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(mark_1)
    x = Add()([x, mark_1])

    x = BilinearUpSampling(target_size=(input_shape[0] // 8, input_shape[1] // 8))(x)
    mark_2 = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
                    kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(mark_2)
    x = Add()([x, mark_2])

    x = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]), name="FCN_HSR")(x)
    output_x = Activation("softmax")(x)

    return Model(input_x, output_x)

def FCN_VGG19_sen(input_shape,
                  n_class,
                  weight_decay=1e-4,
                  kernel_initializer="he_normal"):

    input_x = Input(shape=input_shape)

    x = conv_act_block(input_x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    x = conv_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    mark_2 = x

    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    mark_1 = x

    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = conv_act_block(x, 1024, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    x = conv_act_block(x, 1024, weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    x = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling(target_size=(16, 16))(x)
    mark_1 = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(mark_1)
    x = Add()([x, mark_1])

    x = BilinearUpSampling(target_size=(32, 32))(x)
    mark_2 = Conv2D(n_class, (1, 1), padding="same", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(mark_2)
    x = Add()([x, mark_2])

    x = BilinearUpSampling(target_size=(256, 256), name="FCN_SEN")(x)
    output_x = Activation("softmax")(x)

    return Model(input_x, output_x)

if __name__ == "__main__":

    trainurls = "../train.csv"
    valurls = "../validation.csv"
    hsrimage_dir = "../Train_image"
    senimage_dir = "../Train_image_sen"
    label_dir = "../Train_label"
    hsrimagesize = 256
    hsrchannel = 3
    batch_size = 2
    nb_class = 2

    for monte_iter in range(1, 11):
        train_fnames = [line.strip() for line in open(trainurls, "r", encoding="utf-8")]
        val_fnames = [line.strip() for line in open(valurls, "r", encoding="utf-8")]

        train_gen_hsr = tradata_generator(train_fnames, hsrimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size, shuffle=True)
        val_gen_hsr = valdata_generator(val_fnames, hsrimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size)
        model_fcn = FCN_VGG19(input_shape=(hsrimagesize, hsrimagesize, hsrchannel), n_class=nb_class)
        model_fcn.summary()
        model_fcn.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

        logs_dir = "./logsfile"
        model_name = "FCN_VGG19_gz_" + str(monte_iter) + ".h5"
        loss_name = "FCN_VGG19_gz_" + str(monte_iter) + ".csv"

        # callbacks
        n_callbacks = []
        log_path = os.path.join(logs_dir, "logs")
        ckp_path = os.path.join(logs_dir, "checkpoints")
        loss_path = os.path.join(logs_dir, "loss")
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        if not os.path.exists(ckp_path):
            os.mkdir(ckp_path)
        if not os.path.exists(loss_path):
            os.mkdir(loss_path)
        n_callbacks.append(ModelCheckpoint(ckp_path+"/"+model_name, save_best_only=True, verbose=1))
        n_callbacks.append(TensorBoard(log_dir=log_path))
        n_callbacks.append(LearningRateScheduler(schedule=learning_rate_schedule, verbose=1))

        time_start = time.time()
        H = model_fcn.fit_generator(generator=train_gen_hsr, steps_per_epoch=len(train_fnames)//batch_size, epochs=20, verbose=1, callbacks=n_callbacks,
                                    validation_data=val_gen_hsr, validation_steps=len(val_fnames)//batch_size)
        time_end =time.time()
        print(str(time_end - time_start) + "s")

        # save acc and loss
        loss_df = pd.DataFrame(H.history["loss"], dtype=float).T
        loss_df = loss_df.append(pd.DataFrame(H.history["val_loss"], dtype=float).T)
        loss_df = loss_df.append(pd.DataFrame(H.history["acc"], dtype=float).T)
        loss_df = loss_df.append(pd.DataFrame(H.history["val_acc"], dtype=float).T)
        loss_df.to_csv(loss_path+"/"+loss_name, header=False, index=False, mode="w")