import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Conv2D = functools.partial(layers.Conv2D, activation='relu', padding='same', use_bias=False)
DeConv2D = functools.partial(layers.Conv2DTranspose, activation='relu', padding='same', use_bias=False)

class ResConv(keras.Model):
    def __init__(self, nchannel, pfx=''):
        super(ResConv, self).__init__()

        self.conv1 = Conv2D(nchannel, 3, name=pfx + '_1')
        self.conv2 = Conv2D(nchannel, 3, name=pfx + '_2', activation=None)

    def call(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out = x + out1
        return out

class ResConv_LN(keras.Model):
    def __init__(self, nchannel, pfx=''):
        super(ResConv_LN, self).__init__()

        self.conv1 = Conv2D(nchannel, 3, name=pfx + '_1')
        self.conv2 = Conv2D(nchannel, 3, name=pfx + '_2', activation=None)

        self.ln = layers.LayerNormalization(axis=-1)

    def call(self, x):
        out1 = self.ln(x)
        out1 = self.conv1(out1)
        out1 = self.conv2(out1)
        out = x + out1
        return out


########################################################
#### fast Fourier Convolution-based Feature Enrichment
class FCFE(keras.Model):
    def __init__(self, nchannel, pfx=''):
        super(FCFE, self).__init__()

        self.conv1 = Conv2D(nchannel * 2, 1, name=pfx + '_1')
        self.conv2 = Conv2D(nchannel * 2, 1, name=pfx + '_2', activation=None)

        self.conv3 = Conv2D(nchannel , 1, name=pfx+'_3')

    def call(self, x1, x2):
        fft_x1 = tf.transpose(tf.signal.rfft2d(tf.transpose(x1, [0, 3, 1, 2])), [0, 2, 3, 1])
        real_x1 = tf.math.real(fft_x1)
        imag_x1 = tf.math.imag(fft_x1)
        # mag_x1 = tf.abs(fft_x1)

        fft_x2 = tf.transpose(tf.signal.rfft2d(tf.transpose(x2, [0, 3, 1, 2])), [0, 2, 3, 1])
        real_x2 = tf.math.real(fft_x2)
        imag_x2 = tf.math.imag(fft_x2)
        # mag_x2 = tf.abs(fft_x2)

        out = tf.concat((real_x1, imag_x1, real_x2, imag_x2), axis=3)
        out = self.conv1(out)
        out = self.conv2(out)
        out_real, out_imag = tf.split(out, 2, axis=3)
        out = tf.complex(out_real, out_imag)
        out_f = tf.transpose(tf.signal.irfft2d(tf.transpose(out, [0, 3, 1, 2])), [0, 2, 3, 1])

        out = tf.concat((x1, x2, out_f), axis=3)
        out = self.conv3(out)
        return out





