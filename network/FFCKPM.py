import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from network.network_utils_wob import ResConv_LN, Conv2D, DeConv2D, FCFE

##################################################
### Fast Fourier Convolution-enhanced Kernel Prediction Modeling
class FFCKPM(keras.Model):
    def __init__(self, num_basis=90, ksz=15, burst_length=8, color=False, nchannel=64):
        super(FFCKPM, self).__init__()

        self.encoder = Encoder(nchannel)
        self.basis_decoder = Basis_Decoder(ksz, burst_length, num_basis, color, nchannel)
        self.coeff_decoder = Coeff_Decoder(num_basis, nchannel)

    def call(self, x):
        skips = self.encoder(x)
        coeffs, co_skips = self.coeff_decoder(skips)
        basis = self.basis_decoder(co_skips)
        return basis, coeffs


class Encoder(keras.Model):
    def __init__(self, nchannel, pfx='encoder'):
        super(Encoder, self).__init__()

        self.e1 = EncoderBlock(nchannel, False, pfx=pfx+'_e1')
        self.e2 = EncoderBlock(nchannel*2, pfx=pfx+'_e2')
        self.e3 = EncoderBlock(nchannel*4, pfx=pfx+'_e3')
        self.e4 = EncoderBlock(nchannel*8, pfx=pfx+'_e3')

        self.mlff1 = MLFF(nchannel, pfx=pfx+'_aff1')
        self.mlff2 = MLFF(nchannel, pfx=pfx+'_aff2')


    def call(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        e12 = tf.image.resize(e1, tf.shape(e2)[1:3])
        e32 = tf.image.resize(e3, 2 * tf.shape(e3)[1:3])
        e21 = tf.image.resize(e2, 2 * tf.shape(e2)[1:3])
        e31 = tf.image.resize(e32, 2 * tf.shape(e32)[1:3])

        e2_new = self.mlff2(e12, e2, e32)
        e1_new = self.mlff1(e1, e21, e31)

        return [e4, e3, e2_new, e1_new]


class EncoderBlock(keras.Model):
    def __init__(self, nchannel, stride=True, pfx=''):
        super(EncoderBlock, self).__init__()

        if stride:
            self.conv = Conv2D(nchannel, 3, strides=(2,2), name=pfx+'_1')
        else:
            self.conv = Conv2D(nchannel, 3, name=pfx+'_1')

        self.fftres1 = ResConv_LN(nchannel, pfx=pfx+'_1')

    def call(self, x):
        x = self.conv(x)
        x = self.fftres1(x)

        return x

class MLFF(keras.Model):
    def __init__(self, nchannel, pfx=''):
        super(MLFF, self).__init__()

        self.conv1 = Conv2D(nchannel, 1, name=pfx+'_1')
        self.conv2 = Conv2D(nchannel, 3, activation=None, name=pfx+'_2')

    def call(self, x1, x2, x3):
        x = tf.concat([x1, x2, x3], -1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Coeff_Decoder(keras.Model):
    def __init__(self, num_basis, nchannel, pfx='coeff_decoder'):
        super(Coeff_Decoder, self).__init__()

        self.coeff_de_block1 = CoeffDeBlock(nchannel*8, skip=False, pfx=pfx+'_block1')
        self.coeff_de_block2 = CoeffDeBlock(nchannel*4, pfx=pfx+'_block2')
        self.coeff_de_block3 = CoeffDeBlock(nchannel*2, pfx=pfx+'_block3')
        self.coeff_de_block4 = CoeffDeBlock(nchannel, stride=False, pfx=pfx+'_block4')

        self.conv = Conv2D(num_basis, 3, activation=None, name=pfx+'_end')
        self.softmax = layers.Softmax()

    def call(self, skips):
        d1, s1 = self.coeff_de_block1(skips[0], None) # 16->32
        d2, s2 = self.coeff_de_block2(d1, skips[1]) # 32->64
        d3, s3 = self.coeff_de_block3(d2, skips[2]) # 64->128
        d4, s4 = self.coeff_de_block4(d3, skips[3]) #

        out = self.conv(d4)
        out = self.softmax(out)
        return out, [skips[0], s1, s2, s3, s4]


class CoeffDeBlock(keras.Model):
    def __init__(self, nchannel, skip=True, stride=True, pfx=''):
        super(CoeffDeBlock, self).__init__()

        self.skip = skip
        self.stride = stride
        if self.skip:
            self.conv_skip = FCFE(nchannel, pfx=pfx+'_skipcat')

        self.fftres = ResConv_LN(nchannel, pfx=pfx+'_1')

        if self.stride:
            self.deconv = DeConv2D(nchannel//2, 4, strides=(2,2), name=pfx+'_deconv')

    def call(self, x, skip):
        if self.skip:
            x = self.conv_skip(x, skip)

        skip = self.fftres(x)

        if self.stride:
            x = self.deconv(skip)
        else:
            x = skip
        return x, skip


class Basis_Decoder(keras.Model):
    def __init__(self, ksz, burst_length, num_basis, color, nchannel, pfx='basis_decoder'):
        super(Basis_Decoder, self).__init__()

        self.ksz = ksz
        self.burst_length = burst_length
        self.num_basis = num_basis
        self.color = color

        self.e5 = EncoderBlock(nchannel*8, pfx=pfx+'_e5')

        self.d0 = BasisDeBlock2(nchannel*4, pfx=pfx+'_d0')
        self.d1 = BasisDeBlock2(nchannel*2, pfx=pfx+'_d1')
        self.d2 = BasisDeBlock2(nchannel*2, pfx=pfx+'_d2')
        self.d3 = BasisDeBlock2(nchannel, pfx=pfx+'_d3')

        self.conv1 = Conv2D(128, 2, padding='valid', name='basis_end1')
        self.conv2 = Conv2D(128, 3, name='basis_end2')

        if self.color:
            self.conv3 = Conv2D(
                self.burst_length * 3 * self.num_basis, 3,
                activation=None, name='basis_end3')
            self.reshape = layers.Reshape(
                (self.ksz ** 2 * self.burst_length, 3, self.num_basis))
            self.softmax = layers.Softmax(axis=-3)
            self.reshape2 = layers.Reshape(
                (self.ksz ** 2 * self.burst_length * 3, self.num_basis))
        else:
            self.conv3 = Conv2D(
                self.burst_length * self.num_basis, 3,
                activation=None, name='basis_end3')
            self.reshape = layers.Reshape(
                (self.ksz ** 2 * self.burst_length, self.num_basis))
            self.softmax = layers.Softmax(axis=-2)

    def call(self, skips):
        e5 = self.e5(skips[0]) # 8x8
        x = tf.reduce_mean(e5, axis=[1, 2], keepdims=True)  # 1x1

        x = self.d0(x, skips[1]) # 2x2
        x = self.d1(x, skips[2]) # 4x4
        x = self.d2(x, skips[3]) # 8x8
        x = self.d3(x, skips[4]) # 16x16

        x = self.conv1(x)
        x = self.conv2(x)

        if self.color:
            out = self.conv3(x)
            out = self.reshape(out)
            out = self.softmax(out)
            out = self.reshape2(out)
        else:
            out = self.conv3(x)
            out = self.reshape(out)
            out = self.softmax(out)

        out = tf.transpose(out, [0, 2, 1])
        return out



class BasisDeBlock2(keras.Model):
    def __init__(self, nchannel, pfx=''):
        super(BasisDeBlock2, self).__init__()
        self.deconv = DeConv2D(nchannel, 4, strides=(2,2), name=pfx + '_1')

        self.fft = ResConv_LN(nchannel, pfx=pfx+'_1')

    def call(self, x, skip):
        shape = tf.shape(x)

        skip = tf.reduce_mean(skip, axis=[1, 2], keepdims=True)
        skip = tf.tile(skip, [1, shape[1], shape[2], 1])

        x = tf.concat([x, skip], -1)
        x = self.deconv(x)

        x = self.fft(x)

        return x
