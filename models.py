from GDI.model.base import BaseModel
import tensorflow as tf
from keras.utils import plot_model
from keras.optimizers import RMSprop
import math
import os
import numpy as np
from GDI.utils.generator import BaseGenerator
from keras.initializers import he_normal
from keras.losses import BinaryCrossentropy
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Add, MaxPooling2D, UpSampling2D, concatenate, ReLU, Layer, Input, LeakyReLU
import keras.backend  as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Add, MaxPooling2D, UpSampling2D, concatenate, ReLU, Layer, Input, LeakyReLU
import keras.backend as K

from utils import encode,decode

import tensorflow as tf
from PIL import Image

from conf import data_path


class MultiPose(BaseModel):
    def __init__(self, input_shape, name="ModelA"):
        super().__init__(input_shape, name)

    def __bottleneck(self, input_layer, filters):
        conv1 = Conv2D(kernel_size=1, filters=filters//2, strides=1, padding='same')(input_layer)
        conv1 = LeakyReLU()(conv1)
        conv1 = BatchNormalization()(conv1)

        conv2 = Conv2D(kernel_size=3, filters=filters//2, strides=1, padding='same')(conv1)
        conv2 = LeakyReLU()(conv2)
        conv2 = BatchNormalization()(conv2)

        conv1 = Conv2D(kernel_size=1, filters=filters, strides=1, padding='same')(conv2)
        conv1 = LeakyReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        # model = Model(inputs=input_layer, outputs=Add()([inputs, conv1]))
        return Add()([input_layer, conv1])

    def __hourglass_module(self, input_layer, filter):
        def __side_connection(a, b):
            return Add()([self.__bottleneck(a, filters=filter), b])

        bottle1 = self.__bottleneck(input_layer, filters=filter)
        bottle1 = MaxPooling2D()(bottle1)

        bottle2 = self.__bottleneck(bottle1, filters=filter)
        bottle2 = MaxPooling2D()(bottle2)

        bottle3 = self.__bottleneck(bottle2, filters=filter)
        bottle3 = MaxPooling2D()(bottle3)

        bottle4 = self.__bottleneck(bottle3, filters=filter)
        bottle4 = MaxPooling2D()(bottle4)

        bottle5 = self.__bottleneck(bottle4, filters=filter)
        bottle6 = self.__bottleneck(bottle5, filters=filter)
        bottle7 = self.__bottleneck(bottle6, filters=filter)

        up1 = UpSampling2D()(bottle7)
        up1 = __side_connection(bottle3,up1)

        up2 = UpSampling2D()(up1)
        up2 = __side_connection(bottle2,up2)

        up3 = UpSampling2D()(up2)
        up3 = __side_connection(bottle1,up3)

        up4 = UpSampling2D()(up3)
        up4 = __side_connection(input_layer,up4)

        last_bottle = self.__bottleneck(up4, filters=filter)
        return last_bottle

    def __intermediate(self, bottle_base, out_filter):
        intermediate = Conv2D(kernel_size=1, filters=out_filter, strides=1, padding='same', activation='tanh')(bottle_base)

        conv = Conv2D(kernel_size=3, filters=256, strides=1, padding='same')(intermediate)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        conv = self.__bottleneck(conv, 256)
        return intermediate, conv

    def __last_bottle(self, bottle_base, out_filter):
        last = Conv2D(kernel_size=3, filters=256, strides=1, padding='same')(bottle_base)
        last = LeakyReLU()(last)
        last = BatchNormalization()(last)
        last = Conv2D(kernel_size=1, filters=128, strides=1, padding='same')(last)
        last = LeakyReLU()(last)
        last = BatchNormalization()(last)
        last = Conv2D(kernel_size=1, filters=out_filter, strides=1, padding='same', activation='tanh')(last)
        return last

    def FullHourglass(self, input_layer):
        filter = 256
        hour = self.__hourglass_module(input_layer, filter)
        bottle_base = self.__bottleneck(hour, filter)
        confidence_intermediate, confidence_side = self.__intermediate(bottle_base, 17)
        center_intermediate, center_side = self.__intermediate(bottle_base, 16)

        bottle_origin = self.__bottleneck(bottle_base, filter)

        next_input = Add()([bottle_origin, confidence_side, center_side, input_layer])

        # hour_glass 2
        hour = self.__hourglass_module(next_input, filter)
        bottle_base = self.__bottleneck(hour, filter)
        confidence_last = self.__last_bottle(bottle_base, 17)
        center_last = self.__last_bottle(bottle_base, 16)

        return confidence_intermediate, center_intermediate, confidence_last, center_last

    def __build_model__(self, input_layer):
        conv = Conv2D(kernel_size=7, filters=256, strides=2, padding='same')(input_layer)
        conv = BatchNormalization()(conv)
        conv = LeakyReLU()(conv)

        init_bottle = self.__bottleneck(conv, 256)
        max = MaxPooling2D()(init_bottle)

        cf_inter, cc_inter, cf_last, cc_last = self.FullHourglass(max)
        output_layer = [cf_inter, cc_inter, cf_last, cc_last]

        return output_layer


def data_generator(name_list, batch_size, shuffle=True):
    def __load_image(path):
        return encode(Image.open(path))

        # return (np.asarray(Image.open(path))/255.)

    idxes = np.arange(len(name_list))
    if shuffle:
        np.random.shuffle(idxes)

    root_path = 'dataset/mpii'
    for j in range(math.ceil(len(name_list)/batch_size)):
        x = []
        cf_y = []
        cc_y = []
        for batch_idx in range(0, batch_size):
            y_temp = []
            idx = j*batch_size + batch_idx
            idx = name_list[idxes[idx]]
            x_image = __load_image(root_path+'/input/{0}.png'.format(idx))
            for i in range(0, 17):
                y_temp.append(np.reshape(__load_image(root_path+'/heatmap/{0}/{1}.png'.format(i, idx)), (64,64,1)))

            for i in range(0, 16):
                y_temp.append(np.reshape(__load_image(root_path + '/center_limb/{0}/{1}.png'.format(i, idx)), (64, 64, 1)))
            y_temp = np.asarray(y_temp)
            # y_temp = np.reshape(y_temp,(64,64,31))
            cft = y_temp[0]

            for i in range(1,17):
                cft = np.concatenate([cft, y_temp[i]],axis=2)

            cct = y_temp[17]

            for i in range(18, 33):
                cct = np.concatenate([cct, y_temp[i]], axis=2)

            # y.append(y_temp)
            # for idx in range(0,31):
            #     test = y_temp[idx][:,:]
            #     test = np.asarray((test+ 1) * 125., dtype=np.uint8)
            #     test = np.reshape(test, (64, 64))
            #     image = Image.fromarray(test)
            #     image.save('dataset/mpii/result/t/{0}_limb_{1}.png'.format(batch_size,idx))
            #
            # save_limb(y,'dataset/mpii/result/t/{0}_limb.png'.format(batch_idx))
            # save_heatmap(y,'dataset/mpii/result/t/{0}_heatmap.png'.format(batch_idx))
            cf_y.append(cft)
            cc_y.append(cct)
            x.append(x_image)

        # y = np.asarray(y)
        cf_y = np.asarray(cf_y)
        cc_y = np.asarray(cc_y)
        x = np.asarray(x)
        y = [cf_y, cc_y, cf_y, cc_y]
        yield x, y


def focal_loss(y_true, y_pred):

    alpha = 2
    beta = 4
    epsilon = K.epsilon()

    y_true = (y_true + 1)/2.0
    y_pred = (y_pred + 1)/2.0
    N = tf.reduce_sum(tf.where(tf.equal(y_true,1), y_true, tf.ones_like(y_true)))
    y_pred_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    y_pred_0 = tf.where(tf.not_equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
    y_true_0 = tf.where(tf.not_equal(y_true, 1), y_true, tf.zeros_like(y_true))

    y_pred_0 = K.clip(y_pred_0, epsilon, 1.-epsilon)
    y_pred_1 = K.clip(y_pred_1, epsilon, 1.-epsilon)
    y_true_0 = K.clip(y_true_0, epsilon, 1.-epsilon)

    y_pred_1 = tf.pow((1-y_pred_1), alpha) * tf.log(y_pred_1)
    y_pred_0 = tf.pow((1-y_true_0), beta) * tf.pow(y_pred_0,alpha) * tf.log(1-y_pred_0)
    y_pred_sum = tf.reduce_sum(y_pred_1 + y_pred_0)
    return -y_pred_sum / N

def save_limb(values, path):
    limb_gt = values[:,:,0]
    for k in range(1, 16):
        limb_gt = np.maximum(np.reshape(values[:, :, k], (64,64)), limb_gt)
    limb_gt = decode(limb_gt)
    # limb_gt = np.asarray(((limb_gt+1) * 127.5), dtype=np.uint8)
    limb_gt = np.reshape(limb_gt, (64, 64))
    image = Image.fromarray(limb_gt)
    image.save(path)


def save_heatmap(values, path):
    gt = values[:,:,0]
    for k in range(1, 17):
        gt = np.maximum(np.reshape(values[ :, :, k], (64, 64)), gt)
    gt = decode(gt)
    # gt = np.asarray(((gt+1) * 125.), dtype=np.uint8)
    gt = np.reshape(gt, (64, 64))
    image = Image.fromarray(gt)
    image.save(path)


def create_train_test_set(ratio):
    dataes = os.listdir('dataset/mpii/input')
    indexes = np.arange(len(dataes))
    np.random.shuffle(indexes)
    test_len = int(len(indexes)*ratio)
    train_len = len(indexes) - test_len

    train = []
    test = []
    for i in range(test_len):
        test.append(dataes[indexes[i]].split('/')[-1].split('.')[0])

    for i in range(test_len, test_len + train_len):
        train.append(dataes[indexes[i]].split('/')[-1].split('.')[0])

    print(len(train), len(test))
    with open('train_set.ini', 'w') as op:
        for name in train:
            op.write(name+'\n')

    with open('test_set.ini', 'w') as op:
        for name in test:
            op.write(name+'\n')

    return train, test


def load_train_test_set():
    op = open('train_set.ini', 'r')
    train = []
    train_line = op.readlines()
    for line in train_line:
        train.append(line[:-1])

    op = open('test_set.ini', 'r')
    test = []
    test_line = op.readlines()
    for line in test_line:
        test.append(line[:-1])

    return train, test


# def focal_loss(gamma=2., alpha=.25):
#     """
#     Binary form of focal loss.
#       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
#       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     Usage:
#      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#     def focal_loss_fixed(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred:  A tensor resulting from a sigmoid
#         :return: Output tensor.
#         """
#
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.not_equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#
#         epsilon = K.epsilon()
#         # clip to prevent NaN's and Inf's
#         pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
#         pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
#
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
#                -K.sum((1 - pt_0) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#
#     return focal_loss_fixed

def focal_loss(y_true, y_pred):
    epsilon = K.epsilon()

    def one_value(b, yp, alpha):
        return K.clip(-K.pow(b - b * yp, alpha) * K.log(b * yp), epsilon, 1.-epsilon)

    def no_one_value(b, yt, yp, alpha, beta):
        return K.clip(-K.pow(b - b * yt, beta) * K.pow(b * yp, alpha) * K.log(b - yp), epsilon, 1.-epsilon)

    y_true = (y_true + 1) / 2.0
    y_pred = (y_pred + 1) / 2.0 + epsilon

    base = K.cast(K.equal(y_true, 1), dtype='float32')
    no_one = (base - 1) * -1

    alpha = 2
    beta = 4

    ones = one_value(base, y_pred, alpha)
    no_ones = no_one_value(no_one, y_true, y_pred, alpha, beta)

    return K.sum(ones + no_ones) / K.sum(base)


if __name__ == "__main__":
    data_set_path = "dataset/mpii/result_focal"
    os.makedirs(data_set_path, exist_ok=True)

    train, test = load_train_test_set()
    length = 22400
    t = MultiPose(input_shape=(256,256,3), hournum=2)
    plot_model(t.model, to_file='model.png')
    optimzer = RMSprop(lr=2.5e-4)
    t.compile(optimizer=optimzer, loss=focal_loss, metrics=['mae'])
    # t.summary()
    t.model.load_weights('{0}/129/model.h5'.format(data_set_path))
    # t.compile(optimizer='adam', loss=['mse','mse'], metrics=['mae'])
    epoch = 1
    while True:
        os.makedirs('{1}/{0}'.format(epoch,data_set_path),exist_ok=True)

        for idx, value in enumerate(data_generator(train, 4)):
            print('epoch:', epoch, 'iter: ', idx, t.model.train_on_batch(x=value[0], y=value[1]))

        t.model.save('{1}/{0}/model.h5'.format(epoch, data_set_path))
        for idx, value in enumerate(data_generator(10, 1, length-10, shuffle=False)):
            save_heatmap(value[1][0][0], '{2}/{0}/heatmap_gt_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(value[1][1][0], '{2}/{0}/limb_gt_{1}.png'.format(epoch, idx, data_set_path))

            result = t.model.predict(value[0])
            save_heatmap(result[0][0], '{2}/{0}/base_heatmap_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(result[1][0], '{2}/{0}/base_limb{1}.png'.format(epoch, idx, data_set_path))

            save_heatmap(result[2][0], '{2}/{0}/heatmap_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(result[3][0], '{2}/{0}/limb_{1}.png'.format(epoch, idx, data_set_path))

        for idx, value in enumerate(data_generator(10, 1)):
            save_heatmap(value[1][0][0], '{2}/{0}/tr_heatmap_gt_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(value[1][1][0], '{2}/{0}/tr_limb_gt_{1}.png'.format(epoch, idx, data_set_path))

            result = t.model.predict(value[0])
            save_heatmap(result[0][0], '{2}/{0}/tr_base_heatmap_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(result[1][0], '{2}/{0}/tr_base_limb{1}.png'.format(epoch, idx, data_set_path))

            save_heatmap(result[2][0], '{2}/{0}/tr_heatmap_{1}.png'.format(epoch, idx, data_set_path))
            save_limb(result[3][0], '{2}/{0}/tr_limb_{1}.png'.format(epoch, idx, data_set_path))

        epoch = epoch+1
        # BinaryCrossentropy()
