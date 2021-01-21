from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import load_img, img_to_array

from Net_utils import *
from Encoder import scope_table, build_encoder

import pandas as pd
import numpy as np

import cv2
import os
import time
import gdal




def interp_block(inputs,
                 feature_map_shape,
                 level=1,
                 weight_decay=1e-4,
                 kernel_initializer="he_normal",
                 bn_epsilon=1e-3,
                 bn_momentum=0.99):
    """
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param feature_map_shape: tuple, target shape of feature map.
    :param level: int, default 1.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    ksize = (int(round(float(feature_map_shape[0]) / float(level))),
             int(round(float(feature_map_shape[1]) / float(level))))
    stride_size = ksize

    x = MaxPooling2D(pool_size=ksize, strides=stride_size)(inputs)
    x = Conv2D(512, (1, 1), activation=None,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    x = BilinearUpSampling(target_size=feature_map_shape)(x)

    return x


def pyramid_scene_pooling(inputs,
                          feature_map_shape,
                          weight_decay=1e-4,
                          kernel_initializer="he_normal",
                          bn_epsilon=1e-3,
                          bn_momentum=0.99):
    """ PSP module.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param feature_map_shape: tuple, target shape of feature map.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    interp_block1 = interp_block(inputs, feature_map_shape, level=1,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block2 = interp_block(inputs, feature_map_shape, level=2,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block3 = interp_block(inputs, feature_map_shape, level=3,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block6 = interp_block(inputs, feature_map_shape, level=6,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    return Concatenate()([interp_block1, interp_block2, interp_block3, interp_block6])

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

def PSPNet_hsr(input_shape,
               n_class,
               encoder_name,
               encoder_weights=None,
               weight_decay=1e-4,
               kernel_initializer="he_normal",
               bn_epsilon=1e-3,
               bn_momentum=0.99,
               upscaling_method="bilinear"):
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    features_encoder = encoder.get_layer(scope_table[encoder_name]["pool3"]).output
    feature_map_shape = (int(input_shape[0]/8), int(input_shape[1]/8))

    features = pyramid_scene_pooling(features_encoder, feature_map_shape,
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    features = Concatenate()([features_encoder, features])
    features = Conv2D(512, (3, 3), padding="same", use_bias=False, activation=None,
                      kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    features = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(features)
    features = Activation("relu")(features)

    # upsample
    if upscaling_method == "conv":
        features = bn_act_convtranspose(features, 256, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 256, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 128, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 128, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 64, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 64, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        features = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]), name="upsam_psp_hsr")(features)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)

def PSPNet_sen(input_shape,
               n_class,
               encoder_name,
               encoder_weights=None,
               weight_decay=1e-4,
               kernel_initializer="he_normal",
               bn_epsilon=1e-3,
               bn_momentum=0.99,
               upscaling_method="bilinear"):
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    features_encoder = encoder.get_layer(scope_table[encoder_name]["pool3"]).output
    feature_map_shape = (input_shape[0], input_shape[1])

    features = pyramid_scene_pooling(features_encoder, feature_map_shape,
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    features = Concatenate()([features_encoder, features])
    features = Conv2D(512, (3, 3), padding="same", use_bias=False, activation=None,
                      kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    features = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(features)
    features = Activation("relu")(features)

    # upsample
    if upscaling_method == "conv":
        features = bn_act_convtranspose(features, 256, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 256, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 128, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 128, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 64, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 64, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        features = BilinearUpSampling(target_size=(input_shape[0]*8, input_shape[1]*8), name="upsam_psp_sen")(features)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)

def trainModel_psp_hsr(trainurls,
                       valurls,
                       hsrimage_dir,
                       label_dir,
                       nb_class,
                       epochs=20,
                       batch_size=2,
                       hsrimagesize=256,
                       hsrchannel=3,
                       weight_decay=1e-4,
                       kernel_initializer="he_normal",
                       bn_momentum=0.99,
                       bn_epsilon=1e-3,
                       monte_run_count=1):

    train_fnames = [line.strip() for line in open(trainurls, "r", encoding="utf-8")]
    val_fnames = [line.strip() for line in open(valurls, "r", encoding="utf-8")]

    train_gen_hsr = tradata_generator(train_fnames, hsrimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size, shuffle=True)
    val_gen_hsr = valdata_generator(val_fnames, hsrimage_dir, label_dir, hsrimagesize, hsrimagesize, batch_size)

    model_hsr = PSPNet_hsr(input_shape=(hsrimagesize, hsrimagesize, hsrchannel), n_class=nb_class,
                           encoder_name="resnet_v2_50", encoder_weights=None, weight_decay=weight_decay,
                           kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                           upscaling_method="bilinear")
    model_hsr.summary()
    model_hsr.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    logs_dir = "./logsfile"
    model_name = "PSPNet_Res50_" + str(monte_run_count) + ".h5"
    loss_name = "PSPNet_Res50_" + str(monte_run_count) + ".csv"

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
    H = model_hsr.fit_generator(generator=train_gen_hsr, steps_per_epoch=len(train_fnames)//batch_size, epochs=epochs, verbose=1, callbacks=n_callbacks,
                                validation_data=val_gen_hsr, validation_steps=len(val_fnames)//batch_size)
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
        trainModel_psp_hsr(trainurls=trainurls,
                           valurls=valurls,
                           hsrimage_dir=hsrimage_dir,
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