import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from network.flow_utils import bilinear_warp


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, name=None, padding=1, dilation_rate=1):
        super(Conv2D, self).__init__(name=name)

        self.conv_out = layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      dilation_rate=dilation_rate,
                                      activation=layers.LeakyReLU(0.1))

    def call(self, inputs):
        x = self.conv_out(inputs)

        return x


class DeConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, name=None):
        super(DeConv2D, self).__init__(name=name)

        self.deconv_out = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 name=name)

    def call(self, inputs):
        x = self.deconv_out(inputs)

        return x


'''
Function for calculate cost volumn is borrowed from 
        - https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_costvol.py
        Copyright (c) 2018 Phil Ferriere
        MIT License

        which based on 
        - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py

        Written by Patrick Wieschollek, Copyright Yuxin Wu
        Apache License 2.0
'''


def CostVolumn(c1, warp, search_range, name='cost_volumn'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol


class PredictFlow(layers.Layer):
    def __init__(self, name=None):
        super(PredictFlow, self).__init__()

        self.conv_out = layers.Conv2D(filters=2,
                                      kernel_size=3,
                                      strides=1,
                                      name=name,
                                      padding='same')

    def call(self, inputs):
        return self.conv_out(inputs)


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv3_1 = Conv2D(64, 3, 1, name='conv3_1')
        self.conv3_2 = Conv2D(64, 3, 1, name='conv3_2')
        self.conv3_3 = DeConv2D(32, 4, 2, name='conv3_3')


        self.conv2_1 = Conv2D(32, 3, 1, name='conv2_1')
        self.conv2_2 = Conv2D(32, 3, 1, name='conv2_2')
        self.conv2_3 = DeConv2D(16, 4, 2, name='conv2_3')

        self.conv1_1 = Conv2D(16, 3, 1, name='conv1_1')
        self.conv1_2 = Conv2D(16, 3, 1, name='conv1_2')
        self.conv1_3 = DeConv2D(16, 4, 2, name='conv1_3')

        self.convf = Conv2D(3, 3, 1, name='convf')

    def call(self, x1, x2, x3):
        d3 = self.conv3_1(x3)
        d3 = self.conv3_2(d3)
        d3 = self.conv3_3(d3)
        d2 = self.conv2_1(tf.concat((x2, d3), 3))
        d2 = self.conv2_2(d2)
        d2 = self.conv2_3(d2)

        d1 = self.conv1_1(tf.concat((x1, d2), 3))
        d1 = self.conv1_2(d1)
        d1 = self.conv1_3(d1)

        out = self.convf(d1)
        return out

class DOFE(tf.keras.Model):
    '''
    Modified and inherited from the official pytorch version: https://github.com/NVlabs/PWC-Net/tree/master/PyTorch
    '''

    def __init__(self, max_displacement=4):
        super(DOFE, self).__init__()

        self.conv1a = Conv2D(16, kernel_size=3, strides=2, name='conv1a')
        self.conv1aa = Conv2D(16, kernel_size=3, strides=1, name='conv1aa')
        self.conv1b = Conv2D(16, kernel_size=3, strides=1, name='conv1b')
        self.conv2a = Conv2D(32, kernel_size=3, strides=2, name='conv2a')
        self.conv2aa = Conv2D(32, kernel_size=3, strides=1, name='conv2aa')
        self.conv2b = Conv2D(32, kernel_size=3, strides=1, name='conv2b')
        self.conv3a = Conv2D(64, kernel_size=3, strides=2, name='conv3a')
        self.conv3aa = Conv2D(64, kernel_size=3, strides=1, name='conv3aa')
        self.conv3b = Conv2D(64, kernel_size=3, strides=1, name='conv3b')

        self.LeakyReLU = layers.LeakyReLU(0.1)

        self.conv3_1 = Conv2D(128, kernel_size=3, strides=1)
        self.conv3_2 = Conv2D(96, kernel_size=3, strides=1)
        self.conv3_3 = Conv2D(64, kernel_size=3, strides=1)
        self.conv3_4 = Conv2D(32, kernel_size=3, strides=1)
        self.deconv3 = DeConv2D(2, kernel_size=4, strides=2)
        self.upfeat3 = DeConv2D(2, kernel_size=4, strides=2)

        self.conv2_1 = Conv2D(128, kernel_size=3, strides=1)
        self.conv2_2 = Conv2D(96, kernel_size=3, strides=1)
        self.conv2_3 = Conv2D(64, kernel_size=3, strides=1)
        self.conv2_4 = Conv2D(32, kernel_size=3, strides=1)
        self.deconv2 = DeConv2D(2, kernel_size=4, strides=2)

        self.dc_conv1 = Conv2D(128, kernel_size=3, strides=1, padding=1, dilation_rate=1)
        self.dc_conv3 = Conv2D(64, kernel_size=3, strides=1, padding=2, dilation_rate=2)
        self.dc_conv6 = Conv2D(32, kernel_size=3, strides=1, padding=1, dilation_rate=1)

        self.predict_flow3 = PredictFlow(name='flow3_out')
        self.predict_flow2 = PredictFlow(name='flow2_out')
        self.dc_conv7 = PredictFlow()

        self.denoiser = Decoder()

    def call(self, im1, im2, n_std1, n_std2, is_training=True):
        # im1 = inputs[:, :, :, :3]
        # im2 = inputs[:, :, :, 3:]
        _, h, w, _ = im1.shape

        c11 = self.conv1b(self.conv1aa(self.conv1a(tf.concat((im1, n_std1), 3))))
        c21 = self.conv1b(self.conv1aa(self.conv1a(tf.concat((im2, n_std2), 3))))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))


        ### 3rd flow
        corr3 = CostVolumn(c1=c13, warp=c23, search_range=4)
        x = tf.concat([self.conv3_1(corr3), corr3], 3)
        x = tf.concat([self.conv3_2(x), x], 3)
        x = tf.concat([self.conv3_3(x), x], 3)
        x = tf.concat([self.conv3_4(x), x], 3)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # 2nd flow
        warp2 = bilinear_warp(c22, up_flow3 * 5.0)
        corr2 = CostVolumn(c1=c12, warp=warp2, search_range=3)

        x = tf.concat([corr2, c12, up_flow3, up_feat3], 3)
        x = tf.concat([self.conv2_1(x), x], 3)
        x = tf.concat([self.conv2_2(x), x], 3)
        x = tf.concat([self.conv2_3(x), x], 3)
        x = tf.concat([self.conv2_4(x), x], 3)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv6(self.dc_conv3(self.dc_conv1(x)))
        flow2 = flow2 + self.dc_conv7(x)

        denoise1 = self.denoiser(c11, c12, c13)
        denoise2 = self.denoiser(c21, c22, c23)

        if is_training:
            return [flow3, flow2], [denoise1, denoise2]
        else:
            return 20.0 * tf.image.resize(flow2, (h, w), method=tf.image.ResizeMethod.BILINEAR)