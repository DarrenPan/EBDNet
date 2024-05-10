import tensorflow as tf
from tensorflow import keras

from network.DOFE import DOFE
from network.FFCKPM import FFCKPM
from network.network_utils_wob import ResConv, Conv2D

from network.warp import tf_warp
from utils.tf_utils import estimate_std

class EBDNet(keras.Model):
    def __init__(self, pretrained=True, model_path='', color=False, burst_length=8):
        super(EBDNet, self).__init__()

        self.color = color
        self.FlowNet = DOFE()
        if pretrained:
            ckpt_pt = tf.train.Checkpoint(net=self.FlowNet)
            print('load pretrained flownet from ckpt file.(pretrained in optical flow dataset)')
            ckpt_pt.restore(model_path)
        else:
            print('not use pretrained flownet')

        self.PredictNet = FFCKPM(color=color, burst_length=burst_length)
        self.bl = burst_length

        self.fea_en = keras.Sequential([
            Conv2D(64, 3),
            ResConv(64),
            ResConv(64),
            # ResConv(64),
            Conv2D(64, 3)
        ])
        self.fea_en_ref = keras.Sequential([
            Conv2D(64, 3),
            ResConv(64),
            ResConv(64),
            # ResConv(64),
            Conv2D(64, 3)
        ])



    def call(self, noisy_ref, noisy_oth, noise_std, sig_read, sig_shot):
        '''
        :param noisy_ref: [bs*7, h, w, 3]
        :param noisy_oth: [bs*7, h, w, 3]
        :param noise_std: [bs, h, w, 8]
        :return: basis, coeffs
        '''
        b, h, w, _ = noisy_ref.shape
        fea_dim = 64

        if self.color:
            noise_std_re = tf.reshape(noise_std, (-1, h, w, self.bl, 3))
            noise_std_re = tf.transpose(noise_std_re, perm=[0, 3, 1, 2, 4])
            noise_std_ref = tf.tile(noise_std_re[:, :1, ...], (1, self.bl-1, 1, 1, 1))
            noise_std_oth = noise_std_re[:, 1:, ...]
            noise_std_ref = tf.reshape(noise_std_ref, (-1, h, w, 3))
            noise_std_oth = tf.reshape(noise_std_oth, (-1, h, w, 3))
        else:
            noise_std_exp = tf.expand_dims(noise_std, -1)
            noise_std_exp = tf.transpose(noise_std_exp, perm=[0, 3, 1, 2, 4])
            noise_std_exp_t = tf.tile(noise_std_exp, (1, 1, 1, 1, 3))  # b 8 128 128 3
            noise_std_ref = tf.tile(noise_std_exp_t[:, :1, ...], (1, self.bl-1, 1, 1, 1))
            noise_std_oth = noise_std_exp_t[:, 1:, ...]
            noise_std_ref = tf.reshape(noise_std_ref, (-1, h, w, 3))
            noise_std_oth = tf.reshape(noise_std_oth, (-1, h, w, 3))

        flow, aux_denoises = self.FlowNet(noisy_ref, noisy_oth, noise_std_ref, noise_std_oth)
        flow_final = 20.0 * tf.image.resize(flow[1], (noisy_oth.shape[-3], noisy_oth.shape[-2]),
                                            method=tf.image.ResizeMethod.BILINEAR)

        if not self.color:
            noisy_ref = tf.reshape(noisy_ref, (-1, self.bl-1, h, w, 3))[:,0,:,:,:1] # b h w 1
            noisy_oth = noisy_oth[..., :1] #[bs*7, h, w, 1]

            noise_std_ref = tf.reshape(noise_std_ref, (-1, self.bl-1, h, w, 3))[:,0,:,:,:1]
            noise_std_oth = noise_std_oth[..., :1]
            noisy_ref_in = tf.concat((noisy_ref, noise_std_ref), -1)
            noisy_oth_in = tf.concat((noisy_oth, noise_std_oth), -1)

            noisy_oth_fea = self.fea_en(noisy_oth_in)
            noisy_ref_fea = self.fea_en_ref(noisy_ref_in)
        else:
            noisy_ref = tf.reshape(noisy_ref, (-1, self.bl - 1, h, w, 3))[:, 0, :, :, :] # b h w 3

            noise_std_ref = tf.reshape(noise_std_ref, (-1, self.bl-1, h, w, 3))[:,0,:,:,:]

            noisy_ref_in = tf.concat((noisy_ref, noise_std_ref), -1)
            noisy_oth_in = tf.concat((noisy_oth, noise_std_oth), -1)

            noisy_oth_fea = self.fea_en(noisy_oth_in)
            noisy_ref_fea = self.fea_en_ref(noisy_ref_in)


        noisy_oth_fea_warp = tf_warp(noisy_oth_fea, flow_final)
        noisy_oth_warp = tf_warp(noisy_oth, flow_final)

        aux_denoise_clean, aux_denoise_oth = aux_denoises[0], aux_denoises[1]
        if self.color:
            aux_denoise_clean = tf.reshape(aux_denoise_clean, (-1, self.bl-1, h, w, 3))
            aux_denoise_clean = tf.transpose(aux_denoise_clean, perm=[0, 2, 3, 1, 4])
            aux_denoise_clean = tf.reshape(aux_denoise_clean, (-1, h, w, 21))
            aux_denoise_oth = tf.reshape(aux_denoise_oth, (-1, self.bl-1, h, w, 3))
            aux_denoise_oth = tf.transpose(aux_denoise_oth, perm=[0, 2, 3, 1, 4])
            aux_denoise_oth = tf.reshape(aux_denoise_oth, (-1, h, w, 21))
            aux_denoise = tf.concat([aux_denoise_clean[...,:3], aux_denoise_oth], 3)
        else:
            aux_denoise_clean = tf.reshape(aux_denoise_clean, (-1, self.bl-1, h, w, 3))[...,0]
            aux_denoise_oth = tf.reshape(aux_denoise_oth, (-1, self.bl-1, h, w, 3))[...,0]
            aux_denoise_clean = tf.transpose(aux_denoise_clean, perm=[0, 2, 3, 1])[...,:1]
            aux_denoise_oth = tf.transpose(aux_denoise_oth, perm=[0, 2, 3, 1]) #!!!
            aux_denoise = tf.concat((aux_denoise_clean, aux_denoise_oth), 3)

        if self.color:
            noisy_oth_fea_warp = tf.reshape(noisy_oth_fea_warp, (-1, 7, h, w, fea_dim))
            noisy_oth_fea_warp = tf.transpose(noisy_oth_fea_warp, perm=[0, 2, 3, 1, 4])
            noisy_oth_fea_warp = tf.reshape(noisy_oth_fea_warp, (-1, h, w, fea_dim*7))

            noisy_oth_warp = tf.reshape(noisy_oth_warp, (-1, self.bl - 1, h, w, 3))
            noisy_oth_warp = tf.transpose(noisy_oth_warp, perm=[0, 2, 3, 1, 4])
            noisy_oth_warp = tf.reshape(noisy_oth_warp, (-1, h, w, 21))

            noisy_new = tf.concat([noisy_ref, noisy_oth_warp], 3)
            net_input = tf.concat([noisy_ref_fea, noisy_oth_fea_warp], 3)

        else:

            noisy_oth_fea_warp = tf.reshape(noisy_oth_fea_warp, (-1, self.bl-1, h, w, fea_dim))
            noisy_oth_fea_warp = tf.transpose(noisy_oth_fea_warp, perm=[0, 2, 3, 1, 4])
            noisy_oth_fea_warp = tf.reshape(noisy_oth_fea_warp, (-1, h, w, (self.bl-1)*fea_dim))

            noisy_oth_warp = tf.reshape(noisy_oth_warp, (-1, self.bl - 1, h, w, 1))[..., 0]
            noisy_oth_warp = tf.transpose(noisy_oth_warp, perm=[0, 2, 3, 1])

            noisy_new = tf.concat([noisy_ref, noisy_oth_warp], 3)
            net_input = tf.concat([noisy_ref_fea, noisy_oth_fea_warp], 3)

        noise_std_new = estimate_std(noisy_new, sig_read, sig_shot)
        net_input_wnstd = tf.concat([net_input, noise_std_new], axis=-1)
        basis, coeffs = self.PredictNet(net_input_wnstd)

        return basis, coeffs, noisy_new, aux_denoise