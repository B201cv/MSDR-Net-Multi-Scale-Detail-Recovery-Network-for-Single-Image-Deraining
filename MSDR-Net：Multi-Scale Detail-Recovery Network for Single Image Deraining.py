from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import save_img
# from skimage.measure import compare_psnr, compare_ssim
from keras.layers import Input,PReLU,multiply
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose,AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Multiply, Subtract, concatenate, Add, GlobalAveragePooling2D,Lambda,add
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from PIL import Image
from keras import layers
from scipy.signal import convolve2d
# import argparse

import random
from keras.layers.normalization import BatchNormalization
import os
import cv2
from keras import backend as K
import tensorflow as tf
from keras import initializers
import math
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
# dimensions of our images.
nb_rainy_train_samples = 1800
nb_clean_train_samples = 1800
nb_rainy_test_samples = 200
nb_clean_test_samples = 200
IMG_CHANNELS = 3
IMG_ROWS = 352
IMG_COLS = 352
BATCH_SIZE = 1
NB_EPOCH = 350
NB_SPLIT = 1800
feature_dim=64
input_channel_num=3
nb_filter = 64
growth_rate = 64
nb_layers = 1
# VERBOSE=1
OPTIM_main = Adam(lr=0.00005, beta_1=0.5)
initializers = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
def calc_psnr(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
       h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)  #np.rot90 将矩阵逆时针旋转90°


def calc_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def load_rainy_train_data_heavy():
    Rainy_train_data = np.zeros([nb_rainy_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    Rainy_train_data_r2 = np.zeros([nb_rainy_train_samples, int(IMG_ROWS/2), int(IMG_COLS/2), 3], dtype='float32')
    Rainy_train_data_r3 = np.zeros([nb_rainy_train_samples, int(IMG_ROWS/4), int(IMG_COLS/4), 3], dtype='float32')
    class_path = 'rain_data_train_Light/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split("-")[1].split("x")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split("-")[1].split("x")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img_2 = cv2.resize(img,(int(IMG_ROWS / 2), int(IMG_COLS / 2)))
                img_2 = img_to_array(img_2)
                img_3 = cv2.resize(img,(int(IMG_ROWS / 4), int(IMG_COLS / 4)))
                img_3 = img_to_array(img_3)
                img = img_to_array(img)
                Rainy_train_data[count] = img
                Rainy_train_data_r2[count] = img_2
                Rainy_train_data_r3[count] = img_3
                count = count + 1
    return Rainy_train_data,Rainy_train_data_r2,Rainy_train_data_r3


def load_clean_train_data_heavy():
    Clean_train_data = np.zeros([nb_clean_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    Clean_train_data_r2 = np.zeros([nb_clean_train_samples, int(IMG_ROWS/2), int(IMG_COLS/2), 3], dtype='float32')
    Clean_train_data_r3 = np.zeros([nb_clean_train_samples, int(IMG_ROWS/4), int(IMG_COLS/4), 3], dtype='float32')
    class_path = 'rain_data_train_Light/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split("-")[1].split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split("-")[1].split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img_2 = cv2.resize(img,(int(IMG_ROWS / 2), int(IMG_COLS / 2)))
                img_2 = img_to_array(img_2)
                img_3 = cv2.resize(img,(int(IMG_ROWS / 4), int(IMG_COLS / 4)))
                img_3 = img_to_array(img_3)
                img = img_to_array(img)
                Clean_train_data[count] = img
                Clean_train_data_r2[count] = img_2
                Clean_train_data_r3[count] = img_3
                count = count + 1
    return Clean_train_data,Clean_train_data_r2,Clean_train_data_r3

def load_rainy_test_data_heavy():
    Rainy_test_data = np.zeros([nb_rainy_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    Rainy_test_data_r2 = np.zeros([nb_rainy_train_samples, int(IMG_ROWS / 2), int(IMG_COLS / 2), 3], dtype='float32')
    Rainy_test_data_r3 = np.zeros([nb_rainy_train_samples, int(IMG_ROWS / 4), int(IMG_COLS / 4), 3], dtype='float32')
    class_path = 'rain_data_test_Light/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split("-")[1].split("x")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split("-")[1].split("x")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img_2 = cv2.resize(img, (int(IMG_ROWS / 2), int(IMG_COLS / 2)))
                img_2 = img_to_array(img_2)
                img_3 = cv2.resize(img, (int(IMG_ROWS / 4), int(IMG_COLS / 4)))
                img_3 = img_to_array(img_3)
                img = img_to_array(img)
                Rainy_test_data[count] = img
                Rainy_test_data_r2[count] = img_2
                Rainy_test_data_r3[count] = img_3
                count = count + 1
    return Rainy_test_data,Rainy_test_data_r2,Rainy_test_data_r3


def load_clean_test_data_heavy():
    Clean_test_data = np.zeros([nb_clean_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    Clean_test_data_r2 = np.zeros([nb_clean_train_samples, int(IMG_ROWS / 2), int(IMG_COLS / 2), 3], dtype='float32')
    Clean_test_data_r3 = np.zeros([nb_clean_train_samples, int(IMG_ROWS / 4), int(IMG_COLS / 4), 3], dtype='float32')
    class_path = 'rain_data_test_Light/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split("-")[1].split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split("-")[1].split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img_2 = cv2.resize(img, (int(IMG_ROWS / 2), int(IMG_COLS / 2)))
                img_2 = img_to_array(img_2)
                img_3 = cv2.resize(img, (int(IMG_ROWS / 4), int(IMG_COLS / 4)))
                img_3 = img_to_array(img_3)
                img = img_to_array(img)
                Clean_test_data[count] = img
                Clean_test_data_r2[count] = img_2
                Clean_test_data_r3[count] = img_3
                count = count + 1
    return Clean_test_data,Clean_test_data_r2,Clean_test_data_r3

def load_rainy_train_data_real():
    Rainy_train_data = np.zeros([nb_rainy_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'rain_train_data_real/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Rainy_train_data[count] = img
                count = count + 1
    return Rainy_train_data


def load_clean_train_data_real():
    Clean_train_data = np.zeros([nb_clean_train_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'rain_train_data_real/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Clean_train_data[count] = img
                count = count + 1
    return Clean_train_data


def load_rainy_test_data_real():
    Rainy_test_data = np.zeros([nb_rainy_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'rain_test_data_real/rain/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Rainy_test_data[count] = img
                count = count + 1
    return Rainy_test_data


def load_clean_test_data_real():
    Clean_test_data = np.zeros([nb_clean_test_samples, IMG_ROWS, IMG_COLS, 3], dtype='float32')
    class_path = 'rain_test_data_real/clean/'
    filelists = os.listdir(class_path)
    sort_num_first = []
    for file in filelists:
        sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.sort()
        sorted_file = []
    count = 0
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                img = cv2.imread(class_path + file)
                img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                img = (img - 127.5) / 127.5
                img = img_to_array(img)
                Clean_test_data[count] = img
                count = count + 1
    return Clean_test_data


#
# def RDN_conv(x, nb_filter):
#     x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)
#     x = Activation("relu")(x)
#     return x
#
#
# def RDB(x_in, nb_layers, growth_rate):
#     x = x_in
#     for ii in range(nb_layers):
#         conv = RDN_conv(x, growth_rate)
#         x = concatenate([x, conv], axis=3)
#     x = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(x)
#     x_out = add([x, x_in])
#     return x_out


def Res2Block(x_in, in_num):
    x = x_in
    x = Conv2D(in_num, (1, 1), strides=(1, 1), padding='same')(x)
    x = Activation("relu")(x)
    x_1 = Lambda(lambda x: x[:, :, :, 0:int(in_num/4)])(x)
    x_2 = Lambda(lambda x: x[:, :, :, int(in_num/4):int(in_num/2)])(x)
    x_3 = Lambda(lambda x: x[:, :, :, int(in_num/2):int(in_num/4*3)])(x)
    x_4 = Lambda(lambda x: x[:, :, :, int(in_num/4*3):int(in_num)])(x)
    y_1 = x_1
    y_2 = Conv2D(int(in_num/4), (3, 3), strides=(1, 1), padding='same')(x_2)
    y_2 = Activation("relu")(y_2)
    x_3 = add([y_2, x_3])
    y_3 = Conv2D(int(in_num/4), (3, 3), strides=(1, 1), padding='same')(x_3)
    y_3 = Activation("relu")(y_3)
    x_4 = add([y_3, x_4])
    y_4 = Conv2D(int(in_num/4), (3, 3), strides=(1, 1), padding='same')(x_4)
    y_4 = Activation("relu")(y_4)
    y = concatenate([y_1, y_2, y_3, y_4], axis=3)
    y = Conv2D(in_num, (1, 1), strides=(1, 1), padding='same')(y)
    y = Activation("relu")(y)

    x_a_out = GlobalAveragePooling2D()(y)
    x_fc_1 = Dense(in_num)(x_a_out)
    x_fc_1 = Activation("relu")(x_fc_1)
    x_fc_2 = Dense(in_num)(x_fc_1)
    x_c = Activation("sigmoid")(x_fc_2)
    x_c = Reshape([1, 1, in_num])(x_c)
    x_c_out = Multiply()([x_c, y])
    y_out = add([x_c_out, x])
    return y_out
def rain_branch(inputs,growth_rate):
    inputs = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(inputs)
    inputs = Activation("relu")(inputs)

    x = Res2Block(inputs, growth_rate)
    x = Activation("relu")(x)
    x = Res2Block(x, growth_rate)
    x = Activation("relu")(x)
    se_shape = (1, 1, growth_rate)
    se = GlobalAveragePooling2D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(4, activation="relu",
               kernel_initializer="he_normal", use_bias=False)(se)
    se = Dense(growth_rate, activation="hard_sigmoid",
               kernel_initializer="he_normal", use_bias=False)(se)
    x = multiply([x, se])
    m = Add()([x, inputs])
    return m


def detail_branch(inputs,growth_rate):
    inputs = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(inputs)
    inputs = Activation("relu")(inputs)

    x = _empty_block(inputs,growth_rate)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = _empty_block(x,growth_rate)
    x = BatchNormalization()(x)
    m = Add()([x, inputs])
    return m


def _empty_block(inputs,growth_rate):
    x1 = Conv2D(growth_rate, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x2 = Conv2D(growth_rate, (3, 3), dilation_rate=3, padding="same", kernel_initializer="he_normal")(inputs)
    x3 = Conv2D(growth_rate, (3, 3), dilation_rate=5, padding="same", kernel_initializer="he_normal")(inputs)
    x = concatenate([x1, x2, x3], axis=-1)
    x_out = Conv2D(growth_rate, (1, 1), padding="same", kernel_initializer="he_normal")(x)
    return x_out

def CA(input,growth_rate):
    x_a_out = GlobalAveragePooling2D()(input)
    x_fc_1 = Dense(growth_rate)(x_a_out)
    x_fc_1 = Activation("relu")(x_fc_1)
    x_fc_2 = Dense(growth_rate)(x_fc_1)
    x_c = Activation("sigmoid")(x_fc_2)
    x_c = Reshape([1, 1, growth_rate])(x_c)
    x_out = Multiply()([x_c, input])
    return x_out

def PA(input,growth_rate):
    x = Conv2D(growth_rate, (1, 1), padding="same")(input)
    x = Activation("relu")(x)
    x = Conv2D(growth_rate, (1, 1), padding="same")(x)
    x_1 = Activation("sigmoid")(x)
    x_out = Multiply()([input, x_1])
    return x_out



def FAM(input,growth_rate):

    x = Conv2D(growth_rate,(1,1), padding="same")(input)
    x = Activation("relu")(x)
    x = add([input, x ])

    x_1 = Conv2D(growth_rate,(1,1), padding="same")(x)
    x_1 = CA(x_1,growth_rate)
    x_2 = PA(x_1,growth_rate)

    x_out = add([x, x_2])
    return x_out
def RB(x_in, nb_layers, growth_rate):
    x_in = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(x_in)
    x_in = Activation("relu")(x_in)

    x = x_in
    for ii in range(nb_layers):
        conv = rain_branch(x, growth_rate)
        x = concatenate([x, conv], axis=3)
    x = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(x)
    x_out = add([x, x_in])
    return x_out

def DB(x_in, nb_layers, growth_rate):
    x_in = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(x_in)
    x_in = Activation("relu")(x_in)

    x = x_in
    for ii in range(nb_layers):
        conv = detail_branch(x,growth_rate)
        x = concatenate([x, conv], axis=3)
    x = Conv2D(growth_rate, (1, 1), strides=(1, 1), padding='same')(x)
    x_out = add([x, x_in])
    return x_out

def SubpixelConv2D(scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)

def deconv2d(inputs):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(inputs)
    x = SubpixelConv2D(scale=2)(x)
    x = layers.advanced_activations.PReLU(shared_axes=[1,2])(x)
    return x


def ED_R(x,nb_layers, growth_rate):
    C_1 = Conv2D(growth_rate, (7, 7), padding="same")(x)
    C_1 = Activation("relu")(C_1)
    # C_1 = RB(C_1, nb_layers, growth_rate)
    C_1 = rain_branch(C_1, growth_rate)
    # C_1 = rain_branch(C_1, growth_rate)
    # C_1 = rain_branch(C_1, growth_rate)
    C_1_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_1)  # 352 256

    # C_2 = RB(C_1_down, nb_layers, growth_rate)
    C_2 = rain_branch(C_1_down,growth_rate)
    # C_2 = rain_branch(C_2, growth_rate)
    # C_2 = rain_branch(C_2, growth_rate)
    C_2_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_2)  # 176 128

    # C_3 = RB(C_2_down, nb_layers, growth_rate*2)
    C_3 = rain_branch(C_2_down, growth_rate)
    C_3 = rain_branch(C_3, growth_rate)
    C_3 = rain_branch(C_3, growth_rate)
    C_3_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_3)  # 88 64

    # C_4 = RB(C_3_down, nb_layers, growth_rate*4)
    C_4 = rain_branch(C_3_down, growth_rate*2)
    C_4 = rain_branch(C_4, growth_rate*2)
    C_4 = rain_branch(C_4, growth_rate * 2)
    C_4_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_4)  # 44 32

    # C_5 = RB(C_4_down, nb_layers, growth_rate*4)
    C_5 = rain_branch(C_4_down, growth_rate*4)
    C_5 = rain_branch(C_5, growth_rate*4)
    C_5 = rain_branch(C_5, growth_rate * 4)
    C_5_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_5)  # 22 16

    C_encoder_out = Conv2D(growth_rate * 8, (11, 11), padding="valid")(C_5_down)  # 1*1

    D_1 = UpSampling2D(size=(11, 11))(C_encoder_out)  # 6*6
    # D_1 = Conv2D(growth_rate*8, (3, 3), padding="same")(D_1)
    # D_1 = RB(D_1, nb_layers, growth_rate*4)
    D_1 = rain_branch(D_1, growth_rate*4)
    D_1 = rain_branch(D_1, growth_rate * 4)
    D_1 = rain_branch(D_1, growth_rate * 4)
    D_1 = concatenate([D_1, C_5_down], axis=3)
    # D_1 = Conv2D(growth_rate*4, (3, 3), padding="same")(D_1)
    # D_1 = Activation("relu")(D_1)

    D_2 = UpSampling2D(size=(2, 2))(D_1)  # 6*6
    # D_2 = RB(D_2, nb_layers, growth_rate*4)
    D_2 = rain_branch(D_2, growth_rate * 2)
    D_2 = rain_branch(D_2, growth_rate * 2)
    D_2 = rain_branch(D_2, growth_rate * 2)
    D_2 = concatenate([D_2, C_4_down], axis=3)
    # D_2 = Conv2D(growth_rate*2, (3, 3), padding="same")(D_2)
    # D_2 = Activation("relu")(D_2)

    D_3 = UpSampling2D(size=(2, 2))(D_2)  # 6*6
    # D_3 = RB(D_3, nb_layers, growth_rate*2)
    D_3 = rain_branch(D_3, growth_rate )
    D_3 = rain_branch(D_3, growth_rate * 2)
    D_3 = rain_branch(D_3, growth_rate * 2)
    D_3 = concatenate([D_3, C_3_down], axis=3)
    # D_3 = Conv2D(growth_rate, (3, 3), padding="same")(D_3)
    # D_3 = Activation("relu")(D_3)

    D_4 = UpSampling2D(size=(2, 2))(D_3)  # 96*96
    D_4 = Conv2D(growth_rate, (3, 3), padding="same")(D_4)
    D_4 = Activation("relu")(D_4)
    D_4 = concatenate([D_4, C_2_down], axis=3)
    D_4_out = Conv2D(growth_rate, (3, 3), padding="same")(D_4)
    D_4_out = Activation("relu")(D_4_out)

    D_5 = UpSampling2D(size=(2, 2))(D_4)  # 96*96
    D_5 = Conv2D(growth_rate, (3, 3), padding="same")(D_5)
    D_5 = Activation("relu")(D_5)
    D_5 = concatenate([D_5, C_1_down], axis=3)
    D_5_out = Conv2D(growth_rate, (3, 3), padding="same")(D_5)
    D_5_out = Activation("relu")(D_5_out)

    D_6 = UpSampling2D(size=(2, 2))(D_5)  # 96*96
    D_6 = Conv2D(growth_rate, (3, 3), padding="same")(D_6)
    D_6 = Activation("relu")(D_6)
    return D_4_out,D_5_out,D_6

def ED_D(x,nb_layers, growth_rate):
    C_1 = Conv2D(growth_rate, (7, 7), padding="same")(x)
    C_1 = Activation("relu")(C_1)
    # C_1 = DB(C_1, nb_layers, growth_rate)
    C_1 = detail_branch(C_1, growth_rate)
    # C_1 = detail_branch(C_1, growth_rate)
    # C_1 = detail_branch(C_1, growth_rate)
    C_1_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_1)  # 352 256

    # C_2 = DB(C_1_down, nb_layers, growth_rate)
    C_2 = detail_branch(C_1_down, growth_rate)
    # C_2 = detail_branch(C_2, growth_rate)
    # C_2 = detail_branch(C_2, growth_rate)
    C_2_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_2)  # 176 128

    # C_3 = DB(C_2_down, nb_layers, growth_rate*2)
    C_3 = detail_branch(C_2_down, growth_rate)
    C_3 = detail_branch(C_3, growth_rate)
    C_3 = detail_branch(C_3, growth_rate)
    C_3_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_3)  # 88 64

    # C_4 = DB(C_3_down, nb_layers, growth_rate*4)
    C_4 = detail_branch(C_3_down, growth_rate*2)
    C_4 = detail_branch(C_4, growth_rate*2)
    C_4 = detail_branch(C_4, growth_rate * 2)
    C_4_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_4)  # 44 32

    # C_5 = DB(C_4_down, nb_layers, growth_rate*4)
    C_5 = detail_branch(C_4_down, growth_rate*4)
    C_5 = detail_branch(C_5, growth_rate*4)
    C_5 = detail_branch(C_5, growth_rate * 4)
    C_5_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_5)  # 22 16

    C_encoder_out = Conv2D(growth_rate * 8, (11, 11), padding="valid")(C_5_down)  # 1*1
    # C_encoder_out = FAM(C_encoder_out, growth_rate * 8)

    D_1 = UpSampling2D(size=(11, 11))(C_encoder_out)  # 6*6
    # D_1 = FAM(D_1, growth_rate*8)
    # D_1 = Conv2D(growth_rate, (3, 3), padding="same")(D_1)
    # D_1 = DB(D_1, nb_layers, growth_rate*4)
    D_1 = detail_branch(D_1, growth_rate*4)
    D_1 = detail_branch(D_1, growth_rate * 4)
    D_1 = detail_branch(D_1, growth_rate * 4)
    D_1 = concatenate([D_1, C_5_down], axis=3)

    D_2 = UpSampling2D(size=(2, 2))(D_1) # 6*6
    # D_2 = DB(D_2, nb_layers, growth_rate*4)
    D_2 = detail_branch(D_2, growth_rate * 2)
    D_2 = detail_branch(D_2, growth_rate * 2)
    D_2 = detail_branch(D_2, growth_rate * 2)
    D_2 = concatenate([D_2, C_4_down], axis=3)

    D_3 = UpSampling2D(size=(2, 2))(D_2)  # 6*6
    # D_3 = FAM(D_3, growth_rate)
    # D_3 = DB(D_3, nb_layers, growth_rate)
    D_3 = detail_branch(D_3, growth_rate)
    D_3 = detail_branch(D_3, growth_rate)
    D_3 = detail_branch(D_3, growth_rate)
    D_3 = concatenate([D_3, C_3_down], axis=3)


    D_4 = UpSampling2D(size=(2, 2))(D_3) # 96*96
    D_4 = Conv2D(growth_rate, (3, 3), padding="same")(D_4)
    D_4 = Activation("relu")(D_4)
    D_4 = concatenate([D_4, C_2_down], axis=3)
    D_4_out = Conv2D(growth_rate, (3, 3), padding="same")(D_4)
    D_4_out = Activation("relu")(D_4_out)

    D_5 = UpSampling2D(size=(2, 2))(D_4) # 96*96
    D_5 = Conv2D(growth_rate, (3, 3), padding="same")(D_5)
    D_5 = Activation("relu")(D_5)
    D_5 = concatenate([D_5, C_1_down], axis=3)
    D_5_out = Conv2D(growth_rate, (3, 3), padding="same")(D_5)
    D_5 = Activation("relu")(D_5_out)

    D_6 = UpSampling2D(size=(2, 2))(D_5) # 96*96
    D_6 = Conv2D(growth_rate, (3, 3), padding="same")(D_6)
    D_6 = Activation("relu")(D_6)
    return D_4_out,D_5_out,D_6


inputA = Input(batch_shape=(BATCH_SIZE, IMG_ROWS, IMG_COLS, IMG_CHANNELS), name='inputA')
inputB = Input(batch_shape=(BATCH_SIZE, IMG_ROWS/2, IMG_COLS/2, IMG_CHANNELS), name='inputB')
inputC = Input(batch_shape=(BATCH_SIZE, IMG_ROWS/4, IMG_COLS/4, IMG_CHANNELS), name='inputC')
#
x = Conv2D(growth_rate, (3, 3), padding="same", kernel_initializer="he_normal")(inputA)
x = PReLU(shared_axes=[1, 2])(x)
#
# # x = Res2Block(x,feature_dim)
# C_1 = Conv2D(growth_rate, (7, 7), padding="same")(x)
# C_1 = Activation("relu")(C_1)
# C_1_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_1)#352 416
#
# C_2 = Conv2D(3, (3, 3), padding="same")(C_1_down)
# C_2 = Activation("relu")(C_2)
# C_2_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_2)#176 208

# C_3 = RDB(C_2_down, nb_layers, growth_rate)
# C_3_down = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C_3)#88 104

x_3_up_r,x_2_up_r,x_1_up_r = ED_R(x,nb_layers, growth_rate)
x_3_up_d,x_2_up_d,x_1_up_d = ED_D(x,nb_layers, growth_rate)



x3_r = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_3_up_r)
x3_r = BatchNormalization()(x3_r)
x3_out_r = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x3_r)
x3_out_r = Activation("tanh", name='x3_out_r')(x3_out_r)

x3_d = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_3_up_d)
x3_d = BatchNormalization()(x3_d)
x3_out_d = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x3_d)
x3_out_d = Activation("tanh", name='x3_out_d')(x3_out_d)

x_3_out1 = Subtract()([inputC, x3_out_r])
x_3_out2 = Add()([x_3_out1, x3_out_d])
x3_out = Activation("tanh", name='x3_out')(x_3_out2)

x2_r = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_2_up_r)
x2_r = BatchNormalization()(x2_r)
x2_out_r = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x_2_up_r)
x2_out_r = Activation("tanh", name='x2_out_r')(x2_out_r)

x2_d = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_2_up_d)
x2_d = BatchNormalization()(x2_d)
x2_out_d = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x2_d)
x2_out_d = Activation("tanh", name='x2_out_d')(x2_out_d)

x_2_out1 = Subtract()([inputB, x2_out_r])
x_2_out2 = Add()([x_2_out1, x2_out_d])
x2_out = Activation("tanh", name='x2_out')(x_2_out2)

x1_r = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_1_up_r)
x1_r = BatchNormalization()(x1_r)
x1_out_r = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x1_r)
x1_out_r = Activation("tanh", name='x1_out_r')(x1_out_r)

x1_d = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x_1_up_d)
x1_d = BatchNormalization()(x1_d)
x1_out_d = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x1_d)
x1_out_d = Activation("tanh", name='x1_out_d')(x1_out_d)

x_1_out1 = Subtract()([inputA, x1_out_r])
x_1_out2 = Add()([x_1_out1, x1_out_d])
x1_out = Activation("tanh", name='x1_out')(x_1_out2)

CLEAN_Model = Model(inputs= inputA,outputs = x1_out)
model = Model(inputs=[inputA,inputB,inputC ],outputs=[x1_out,x2_out,x3_out,x1_out_r,x2_out_r,x3_out_r])


# model = Model(inputs=inputA, outputs=[RRN_out, C_out])
model.summary()
model.compile(loss={'x1_out': "mean_absolute_error",
                    'x2_out': "mean_absolute_error",
                    'x3_out': "mean_absolute_error",
                    'x1_out_r': "mean_absolute_error",
                    'x2_out_r': "mean_absolute_error",
                    'x3_out_r': "mean_absolute_error"
                    },
              loss_weights={'x1_out': 1,
                            'x2_out': 0.01,
                            'x3_out': 0.01,
                            'x1_out_r': 0.1,
                            'x2_out_r': 0.01,
                            'x3_out_r': 0.01},
              optimizer=OPTIM_main)
# model.compile(loss= "mean_absolute_error",optimizer=OPTIM_main)

Rainy_train_data_heavy,Rainy_train_data_heavy_r2,Rainy_train_data_heavy_r3 = load_rainy_train_data_heavy()
Clean_train_data_heavy,Clean_train_data_heavy_r2,Clean_train_data_heavy_r3 = load_clean_train_data_heavy()
Rainy_test_data_heavy ,Rainy_test_data_heavy_r2,Rainy_test_data_heavy_r3= load_rainy_test_data_heavy()
Clean_test_data_heavy,Clean_test_data_heavy_r2,Clean_test_data_heavy_r3 = load_clean_test_data_heavy()

for i in range(NB_EPOCH):
    length_heavy = Rainy_train_data_heavy.shape[0]
    idx_heavy = np.arange(0, length_heavy)
    np.random.shuffle(idx_heavy)
    Rainy_train_data_heavy = Rainy_train_data_heavy[idx_heavy]
    Rainy_train_data_heavy_r2 = Rainy_train_data_heavy_r2[idx_heavy]
    Rainy_train_data_heavy_r3 = Rainy_train_data_heavy_r3[idx_heavy]
    Clean_train_data_heavy = Clean_train_data_heavy[idx_heavy]
    Clean_train_data_heavy_r2 = Clean_train_data_heavy_r2[idx_heavy]
    Clean_train_data_heavy_r3 = Clean_train_data_heavy_r3[idx_heavy]
    Rain_train_image_heavy = np.array_split(Rainy_train_data_heavy, NB_SPLIT)
    Rain_train_image_heavy_r2 = np.array_split(Rainy_train_data_heavy_r2, NB_SPLIT)
    Rain_train_image_heavy_r3 = np.array_split(Rainy_train_data_heavy_r3, NB_SPLIT)
    Clean_train_image_heavy = np.array_split(Clean_train_data_heavy, NB_SPLIT)
    Clean_train_image_heavy_r2 = np.array_split(Clean_train_data_heavy_r2, NB_SPLIT)
    Clean_train_image_heavy_r3 = np.array_split(Clean_train_data_heavy_r3, NB_SPLIT)
    index = list(range(NB_SPLIT))
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
    for j in index:
        Rain_streaks = np.subtract(Rain_train_image_heavy[j], Clean_train_image_heavy[j])
        # print(Rain_streaks.shape)
        # # with tf.Session() as sess: Rain_streaks = Rain_streaks.eval()
        Rain_streaks_3 = np.subtract(Rain_train_image_heavy_r3[j], Clean_train_image_heavy_r3[j])
        Rain_streaks_2 = np.subtract(Rain_train_image_heavy_r2[j], Clean_train_image_heavy_r2[j])
        history = model.train_on_batch(x={'inputA': Rain_train_image_heavy[j],'inputB':Rain_train_image_heavy_r2[j],'inputC':Rain_train_image_heavy_r3[j]},
                                       y={'x1_out': Clean_train_image_heavy[j],
                                           'x2_out': Clean_train_image_heavy_r2[j],
                                           'x3_out': Clean_train_image_heavy_r3[j],
                                           'x1_out_r': Rain_streaks,
                                           'x2_out_r': Rain_streaks_2,
                                           'x3_out_r': Rain_streaks_3})
        # history = model.train_on_batch(x={'inputs': Rain_train_image_heavy[j]},y={'x1_out': Clean_train_image_heavy[j]})
        print('epoch:' + str(i) + '...' +
              'Loss:' + str(history[0]) + '...' +
              'x1_out:' + str(history[1]) + '...' +
              'x2_out:' + str(history[2])+ '...' +
              'x3_out:' + str(history[3])+ '...' +
              'x1_out_r:' + str(history[4])+ '...' +
              'x2_out_r:' + str(history[5])+ '...' +
              'x3_out_r:' + str(history[6]))
        # print('epoch:' + str(i) + '...' +'Loss:' + str(history))

    # [predicted_clean_data_heavy,out1,out2,out3,out4,out5] = model.predict(Rainy_test_data_heavy,Rainy_test_data_heavy_r2,Rainy_test_data_heavy_r3)
    predicted_clean_data_heavy = CLEAN_Model.predict(Rainy_test_data_heavy, batch_size=BATCH_SIZE)
    num = predicted_clean_data_heavy.shape[0]
    Score_psnr_heavy = []
    Score_ssim_heavy = []
    for k in range(num):
        predicted_clean_image_heavy = predicted_clean_data_heavy[k].reshape(IMG_ROWS, IMG_COLS, 3)
        predicted_clean_image_heavy = np.uint8((predicted_clean_image_heavy + 1) * 127.5)

        Clean_test_image_heavy = Clean_test_data_heavy[k]
        Clean_test_image_heavy = Clean_test_image_heavy.reshape(IMG_ROWS, IMG_COLS, 3)
        Clean_test_image_heavy = np.uint8((Clean_test_image_heavy + 1) * 127.5)

        img_PSNR_heavy = calc_psnr(Clean_test_image_heavy, predicted_clean_image_heavy)
        img_SSIM_heavy = calc_ssim(Clean_test_image_heavy, predicted_clean_image_heavy)

        if i%50==0 and k % 10 == 0:
            Image_heavy = np.concatenate((predicted_clean_image_heavy, Clean_test_image_heavy), axis=1)
            cv2.imwrite('output1/' + str(i) + '_' + str(k) + '_' + str(img_PSNR_heavy)+ '_'+str(img_SSIM_heavy)+'.png',
                        Image_heavy)
        Score_psnr_heavy.append(img_PSNR_heavy)
        Score_ssim_heavy.append(img_SSIM_heavy)

    Score_psnr_mean_heavy = np.mean(Score_psnr_heavy)
    Score_ssim_mean_heavy = np.mean(Score_ssim_heavy)
    line_PSNR_heavy = "%.4f \n" % (Score_psnr_mean_heavy)
    with open('output1/PSNR_200L-0314.txt', 'a') as f:
        f.write(line_PSNR_heavy)
    line_SSIM_hevay = "%.4f \n" % (Score_ssim_mean_heavy)
    with open('output1/SSIM_200L-0314.txt', 'a') as f:
        f.write(line_SSIM_hevay)


model.save('output1/BasicL_0314.h5')
