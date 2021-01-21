from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import load_img, img_to_array

from Net_utils import *
from FCN_VGG19 import FCN_VGG19
from U_Net_CPD import U_Net_CPD_sen
from DeepLabv3p import Deeplab_v3p_sen
from FCN_VGG19 import FCN_VGG19_sen
from PSPNet import PSPNet_sen

import pandas as pd
import numpy as np

import cv2
import os
import time
import gdal

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

def tradata_generator_multi(base_urls, traimage_dir, senimage_dir, tralabel_dir, image_width, image_height, batch_size, shuffle=True):
    while True:
        if shuffle:
            np.random.shuffle(base_urls)
        images = []
        senimages = []
        labels = []
        batch_i = 0
        for i in range(len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            img = load_image_gdal(os.path.join(traimage_dir, base_url + ".tif"), mode="color")
            senimg = load_image_gdal(os.path.join(senimage_dir, base_url + ".tif"), mode="color")
            label = load_image_gdal(os.path.join(tralabel_dir, base_url + ".tif"), mode="gray")

            label = np.where(label>=2, 0, label)
            label = to_categorical(label, 2)
            images.append(img)
            senimages.append(senimg)
            labels.append(label)
            if batch_i == batch_size:
                train_data = np.array(images)
                train_sendata = np.array(senimages)
                train_label = np.array(labels)
                yield([train_data, train_sendata], train_label)
                images = []
                senimages = []
                labels = []
                batch_i = 0

def valdata_generator_multi(base_urls, valimage_dir, valsenimage_dir, vallabel_dir, image_width, image_height, batch_size):
    while True:
        images = []
        senimages = []
        labels = []
        batch_i = 0
        for i in range(len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            img = load_image_gdal(os.path.join(valimage_dir, base_url + ".tif"), mode="color")
            senimg = load_image_gdal(os.path.join(valsenimage_dir, base_url + ".tif"), mode="color")
            label = load_image_gdal(os.path.join(vallabel_dir, base_url + ".tif"), mode="gray")

            label = np.where(label>=2, 0, label)
            label = to_categorical(label, 2)
            images.append(img)
            senimages.append(senimg)
            labels.append(label)
            if batch_i == batch_size:
                val_data = np.array(images)
                val_sendata = np.array(senimages)
                val_label = np.array(labels)
                yield ([val_data, val_sendata], val_label)
                images = []
                senimages = []
                labels = []
                batch_i = 0

def learning_rate_schedule(_epoch):
    base_learning_rate = 1e-4
    epoch = 100
    lr_base = base_learning_rate
    lr = lr_base * ((1 - float(_epoch) / epoch) ** 0.9)
    return lr

def trainModel_DS_Net(trainurls,
                      valurls,
                      hsrimage_dir,
                      senimage_dir,
                      label_dir,
                      nb_class,
                      epochs=20,
                      batch_size=2,
                      hsrimagesize=256,
                      hsrchannel=3,
                      senimagesize=32,
                      senchannel=4,
                      weight_decay=1e-4,
                      kernel_initializer="he_normal",
                      bn_momentum=0.99,
                      bn_epsilon=1e-3,
                      monte_run_count=1):

    train_fnames = [line.strip() for line in open(trainurls, "r", encoding="utf-8")]
    val_fnames = [line.strip() for line in open(valurls, "r", encoding="utf-8")]

    model_hsr = FCN_VGG19(input_shape=(hsrimagesize, hsrimagesize, hsrchannel), n_class=nb_class)
    model_sen = PSPNet_sen(input_shape=(senimagesize, senimagesize, senchannel), n_class=nb_class,
                           encoder_name="resnet_v2_50_nosam", encoder_weights=None, weight_decay=weight_decay,
                           kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                           upscaling_method="bilinear")

    V3P_hsr_mid = model_hsr.get_layer("FCN_HSR").output
    V3P_sen_mid = model_sen.get_layer("upsam_psp_sen").output

    V3P_hsr2sen_con = Concatenate()([V3P_hsr_mid, V3P_sen_mid])

    output = Conv2D(nb_class, (1, 1), activation=None, kernel_regularizer=l2(weight_decay),
                    kernel_initializer=kernel_initializer)(V3P_hsr2sen_con)

    output = Activation("softmax")(output)

    DS_V3PNet = Model(inputs=[model_hsr.input, model_sen.input], outputs=output)
    DS_V3PNet.summary()
    DS_V3PNet.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    logs_dir = "./logsfile"
    model_name = "DS_FCN_PSP_sz_" + str(monte_run_count) + ".h5"
    loss_name = "DS_FCN_PSP_sz_" + str(monte_run_count) + ".csv"

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
    train_gen_multi = tradata_generator_multi(train_fnames, hsrimage_dir, senimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size, shuffle=True)
    val_gen_multi = valdata_generator_multi(val_fnames, hsrimage_dir, senimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size)
    H = DS_V3PNet.fit_generator(generator=train_gen_multi, steps_per_epoch=len(train_fnames)//batch_size, epochs=epochs, verbose=2, callbacks=n_callbacks,
                                validation_data=val_gen_multi, validation_steps=len(val_fnames)//batch_size)
    time_end =time.time()
    print(str(time_end - time_start) + "s")

    # save acc and loss
    loss_df = pd.DataFrame(H.history["loss"], dtype=float).T
    loss_df = loss_df.append(pd.DataFrame(H.history["val_loss"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["acc"], dtype=float).T)
    loss_df = loss_df.append(pd.DataFrame(H.history["val_acc"], dtype=float).T)
    loss_df.to_csv(loss_path+"/"+loss_name, header=False, index=False, mode="w")


if __name__ == "__main__":

    trainurls = "../train.csv"
    valurls = "../validation.csv"
    hsrimage_dir = "../Train_image"
    senimage_dir = "../Train_image_sen"
    label_dir = "../Train_label"

    for monte_iter in range(1, 11):
        trainModel_DS_Net(trainurls=trainurls,
                          valurls=valurls,
                          hsrimage_dir=hsrimage_dir,
                          senimage_dir=senimage_dir,
                          label_dir=label_dir,
                          nb_class=2,
                          epochs=20,
                          batch_size=2,
                          hsrimagesize=256,
                          hsrchannel=3,
                          weight_decay=1e-4,
                          kernel_initializer="he_normal",
                          bn_momentum=0.99,
                          bn_epsilon=1e-3,
                          monte_run_count=monte_iter)