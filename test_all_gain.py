#!/usr/bin/env python3

import utils.tf_utils as tfu
from network.EBDNet import EBDNet

import tensorflow as tf
import numpy as np
import time
import argparse
import os
from skimage.metrics import structural_similarity as SSIM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--color', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--psnr', default=True, action='store_true')
parser.add_argument('--ssim', action='store_true')
opts = parser.parse_args()

cal_ssim = opts.ssim
cal_psnr = opts.psnr
save_results = opts.save

gain_ssim_dict = {}
gain_psnr_dict = {}
for gain in [1, 2, 4, 8]:
    if opts.color:
        print('color testing...')
        data = np.load('data/color_testset/%s.npz'%gain)
        noisy_bursts = data['noisy']
        cleans = data['truth']
        white_levels = data['white_level']
        sig_reads = data['sig_read']
        sig_shots = data['sqrt_sig_shot']

        model_path = 'experiments/ebdnet/color/model.ckpt-1400000'

        bsz = 1
    else:
        print('grayscale testing...')

        data = np.load('data/synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz')
        split = {1: 2, 2: 3, 4: 4, 8: 5}[gain]
        noisy_bursts = data['noisy'][73 * split:73 * split + 73].astype(np.float32)
        cleans = data['truth'][73 * split:73 * split + 73].astype(np.float32)
        white_levels = np.ones([73])
        sig_reads = data['sig_read'][73 * split:73 * split + 73].astype(np.float32)
        sig_shots = data['sig_shot'][73 * split:73 * split + 73].astype(np.float32)

        model_path = 'experiments/ebdnet/grayscale/model.ckpt-1000000'
        bsz = 1

    model = EBDNet(pretrained=False, color=opts.color)

    optimizer = tf.keras.optimizers.Adam()
    ckpt_best = tf.train.Checkpoint(model=model)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if model_path:
        ckpt.restore(model_path)
        print("Model restored from " + model_path)
    else:
        print("No trained model is restored.")

    psnrs = []
    ssims = []
    for k in range(sig_reads.shape[0]):

        clean = cleans[k]
        noisy = noisy_bursts[k]
        h, w = noisy.shape[0:2]

        sig_read, sig_shot, white_level = sig_reads[k], sig_shots[k], white_levels[k]

        start_time = time.time()

        # pad image
        scale_size = 16
        h_tmp = h // scale_size * scale_size - h
        h_pad = 0 if h_tmp == 0 else h_tmp + scale_size
        w_tmp = w // scale_size * scale_size - w
        w_pad = 0 if w_tmp == 0 else w_tmp + scale_size
        h_pad, w_pad = np.int32(h_pad), np.int32(w_pad)

        if opts.color:
            noisy = tf.reshape(noisy, [h, w, -1])
        noisy = tf.pad(noisy, [[0, h_pad], [0, w_pad], [0, 0]])
        noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)

        h_new, w_new, _ = noisy.shape

        if not opts.color:
            noisy_exp = tf.expand_dims(noisy, -1)
            noisy_exp = tf.expand_dims(noisy_exp, 0)
            noisy_exp = tf.transpose(noisy_exp, perm=[0, 3, 1, 2, 4])
            noisy_exp = tf.tile(noisy_exp, (1, 1, 1, 1, 3))
            noisy_ref = tf.tile(noisy_exp[:, :1, ...], (1, 7, 1, 1, 1))
            noisy_oth = noisy_exp[:, 1:, ...]

        else:
            noisy_exp = tf.expand_dims(noisy, 0)
            noisy_exp = tf.reshape(noisy_exp, (-1, h_new, w_new, 8, 3))
            noisy_exp = tf.transpose(noisy_exp, perm=[0, 3, 1, 2, 4])
            noisy_ref = tf.tile(noisy_exp[:, :1, ...], (1, 7, 1, 1, 1))
            noisy_oth = noisy_exp[:, 1:, ...]

        noisy_ref = tf.reshape(noisy_ref, (-1, h_new, w_new, 3))
        noisy_oth = tf.reshape(noisy_oth, (-1, h_new, w_new, 3))

        noise_std = tf.expand_dims(noise_std, 0)

        basis, coeffs, noisy_new, aux_denoise = model(noisy_ref, noisy_oth, noise_std, sig_read, sig_shot)

        if opts.color:
            denoise = tfu.apply_filtering_color(noisy_new, basis, coeffs)
            denoise = denoise[:, :h, :w, :]
        else:
            denoise = tfu.apply_filtering_gray(noisy_new, basis, coeffs)
            denoise = denoise[:, :h, :w]

        clean = tfu.restore_and_gamma(
            clean[..., None], white_level).numpy().squeeze()
        denoise = tfu.restore_and_gamma(denoise, white_level).numpy().squeeze()


        def save_img(img, img_path):
            img = img * 255.0
            img = tf.cast(img, dtype=tf.uint8)
            img = tf.image.encode_png(img)
            with tf.io.gfile.GFile(img_path, 'wb') as file:
                file.write(img.numpy())

        if save_results:
            save_denoise = np.expand_dims(np.clip(denoise, 0., 1.), -1)
            save_img(save_denoise, 'results/grayscale/{}/{}_EBDNet.png'.format(str(gain), str(k)))

        # crop this out when reporting psnr, following Mildenhall et al.
        lbuff = 8
        clean = np.clip(clean, 0., 1.)[lbuff:-lbuff, lbuff:-lbuff]
        denoise = np.clip(denoise, 0., 1.)[lbuff:-lbuff, lbuff:-lbuff]

        if cal_ssim:
            if opts.color:
                ssim = SSIM(clean, denoise, gaussian_weights=True, data_range=1.0,  use_sample_covariance=False, multichannel=True)
            else:
                ssim = SSIM(clean, denoise, gaussian_weights=True, data_range=1.0,  use_sample_covariance=False)
        else:
            ssim = 0
        ssims.append(ssim)

        if cal_psnr:
            mse = np.mean(np.square(denoise - clean))
            psnr = np.mean(-10. * np.log10(mse))
        else:
            psnr = 0
        psnrs.append(psnr)

    gain_ssim_dict[gain] = ssims
    gain_psnr_dict[gain] = psnrs

if cal_psnr:
    for gain, psnrs in gain_psnr_dict.items():
        print('Average PSNR: {:.2f} of Gain: {}'.format(np.mean(psnrs), gain))

if cal_ssim:
    for gain, ssims in gain_ssim_dict.items():
        print('Average SSIM: {:.3f} of Gain: {}'.format(np.mean(ssims), gain))
