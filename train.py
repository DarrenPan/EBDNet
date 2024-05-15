#!/usr/bin/env python3

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.EBDNet import EBDNet
from dataset import dataset
import utils.utils as ut
import utils.tf_utils as tfu


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--color', action='store_true')
opts = parser.parse_args()

TLIST = 'dataset/train.txt'
VLIST = 'dataset/val_500.txt'

if opts.color:
    BSZ = 12
    MAXITER = 14e5
    boundaries = [6e5, 8e5]
else:
    BSZ = 6 # 12 for 2 GPUs
    MAXITER = 10e5
    boundaries = [5e5, 7e5]

IMSZ = 128
LR = 1e-4

if opts.debug:
    VALFREQ = 100
    MAXITER = 200
    BSZ = 2
else:
    VALFREQ = 1e5

model_name = 'ebdnet'

exp_root = 'experiments/' + model_name + '/'
if not os.path.exists(exp_root):
    os.makedirs(exp_root)
WTS = exp_root + 'color' if opts.color else exp_root + 'grayscale'
if not os.path.exists(WTS):
    os.makedirs(WTS)
log_writer = ut.LogWriter(WTS + '/train.log')

train_log_dir = exp_root + 'logs/'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# distributed training strategy
strategy = tf.distribute.MirroredStrategy()
ngpus = strategy.num_replicas_in_sync
GLOBAL_BSZ = BSZ * ngpus
log_writer.log("Using %d GPUs." % ngpus)

# learning rate schedule
values = [LR, np.float32(LR / np.sqrt(10.)), LR / 10.]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

with strategy.scope():

    model_path = 'experiments/optical_flow/tf_ckpt/ckpt-400000'
    model = EBDNet(model_path=model_path, color=opts.color)

    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    iterations = optimizer.iterations

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, directory=WTS, checkpoint_name='model.ckpt', max_to_keep=None)

    ckpt_best = tf.train.Checkpoint(model=model)
    manager_best = tf.train.CheckpointManager(ckpt_best, directory=WTS, checkpoint_name='model_best.ckpt', max_to_keep=2)

    resume = False
    if resume:
        ckpt.restore(WTS+'/model.ckpt-XXXXXX')
        log_writer.log("Restored model.")
    else:
        log_writer.log("No previous checkpoints, new training.")

log_writer.log("Creating dataset.")
train_set, val_set = dataset.create_dataset(
    iterations.numpy(), TLIST, VLIST, bsz=GLOBAL_BSZ, repeats=1,
    patches_per_img=1, height=IMSZ, width=IMSZ, grayscale=(not opts.color))
train_dist_set = strategy.experimental_distribute_dataset(train_set)
val_dist_set = strategy.experimental_distribute_dataset(val_set)

def _one_step(inputs, training=True):

    clean, white_level = tfu.degamma_and_scale(inputs)
    noisy, sig_read, sig_shot = tfu.add_read_shot_noise(clean)
    noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)

    if opts.color:
        noisy_re = tf.reshape(noisy, (-1, IMSZ, IMSZ, 8, 3))
        noisy_re = tf.transpose(noisy_re, perm=[0, 3, 1, 2, 4])
        noisy_ref = tf.tile(noisy_re[:, :1, ...], (1, 7, 1, 1, 1))
        noisy_oth = noisy_re[:, 1:, ...]
    else:
        noisy_exp = tf.expand_dims(noisy, -1)
        noisy_exp = tf.transpose(noisy_exp, perm=[0, 3, 1, 2, 4])
        noisy_exp = tf.tile(noisy_exp, (1, 1, 1, 1, 3))  # b 8 128 128 3
        noisy_ref = tf.tile(noisy_exp[:, :1, ...], (1, 7, 1, 1, 1))
        noisy_oth = noisy_exp[:, 1:, ...]

    noisy_ref = tf.reshape(noisy_ref, (-1, IMSZ, IMSZ, 3))
    noisy_oth = tf.reshape(noisy_oth, (-1, IMSZ, IMSZ, 3))

    aux_denosie_loss = 0
    aux_denosie_gradient_loss = 0
    use_framewiseloss = True
    if training:
        with tf.GradientTape() as tape:
            basis, coeffs, noisy_new, aux_denosie = model(noisy_ref, noisy_oth, noise_std, sig_read, sig_shot)

            aux_denosie_loss = tfu.l1_loss(clean, aux_denosie) / ngpus
            aux_denosie_gradient_loss = tfu.gradient_loss(clean, aux_denosie) / ngpus

            if opts.color:
                denoise, framewise = tfu.apply_filtering_color(
                    noisy_new, basis, coeffs, True)
                clean = tfu.restore_and_gamma(clean[...,:3], white_level)
            else:
                denoise, framewise = tfu.apply_filtering_gray(
                    noisy_new, basis, coeffs, True)
                clean = tfu.restore_and_gamma(clean[...,:1], white_level)

            denoise = tfu.restore_and_gamma(denoise, white_level)
            l1_loss = tfu.l1_loss(denoise, clean) / ngpus
            gradient_loss = tfu.gradient_loss(denoise, clean) / ngpus

            if use_framewiseloss:
                framewise = tfu.restore_and_gamma(framewise, white_level[..., None])
                frame_loss, anneal = tfu.frame_loss_2(framewise, clean, iterations)
                frame_loss = frame_loss / ngpus
            else:
                frame_loss = 0

            loss = l1_loss + gradient_loss + \
                   aux_denosie_loss + aux_denosie_gradient_loss + frame_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    else:
        pass


    lvals = {
        'loss': loss,
        'pixel_l1': l1_loss,
        'gradient_l1': gradient_loss,
        'aux_denosie_l1_loss': aux_denosie_loss,
        'aux_denosie_gradient_loss': aux_denosie_gradient_loss,
        'frame_loss': frame_loss,
    }

    return lvals


@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(
    _one_step, args=(dataset_inputs, True))
  return tfu.custom_replica_reduce(
    strategy, tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_val_step(dataset_inputs):
  per_replica_losses = strategy.run(
    _one_step, args=(dataset_inputs, False))
  return tfu.custom_replica_reduce(
    strategy, tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

cur_psnr = 0
best_ckpt = 0
stop = ut.getstop()
log_writer.log("Start training.")
for batch in train_dist_set:
    out = distributed_train_step(batch)

    if iterations.numpy() <= 5e5:
        SAVEFREQ = 1e5
    else:
        SAVEFREQ = 2e4

    with train_summary_writer.as_default():
        for k,v in out.items():
            tf.summary.scalar(k, v, step=iterations.numpy())

    if iterations.numpy() % 100 == 0:
        out = {k + '.t': v for k, v in out.items()}
        out['lr'] = optimizer._decayed_lr('float32').numpy()
        log_writer.log(out, iterations.numpy())


    if iterations.numpy() % SAVEFREQ == 0 and iterations.numpy() != 0:
        log_writer.log("Saving model")
        manager.save(checkpoint_number=iterations.numpy())

    if stop[0] or iterations.numpy() >= MAXITER:
        break

# Save model and optimizer state.
if iterations.numpy() > 0:
    log_writer.log("Saving model and optimizer.")
    manager.save(checkpoint_number=iterations.numpy())
log_writer.log("Stopping!")



