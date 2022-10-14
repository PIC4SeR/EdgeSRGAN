import numpy as np
import tensorflow as tf
import math
from scipy.stats import norm

from utils.niqe import niqe

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

import torch
import lpips

# ---------------------------------------
#  Evaluation
# ---------------------------------------

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch, _ = model(lr_batch, training=False)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    
    return sr_batch


def evaluate(model, dataset, metric='PSNR', perc=None):
    metric_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        if metric == 'PSNR':
            value = tf.image.psnr(hr, sr, max_val=255)[0]
        elif metric == 'NIQE':
            value = tf.map_fn(niqe, sr)
        elif metric == 'SSIM':
            value = tf.image.ssim(hr, sr, max_val=255)[0]   
        elif metric == 'LPIPS':
            with torch.no_grad():
                value = perc(lpips.im2tensor(hr[0].numpy()).cuda(),
                                       lpips.im2tensor(sr[0].numpy()).cuda()).item()                
        metric_values.append(value)
    return tf.reduce_mean(metric_values)



# ---------------------------------------
#  Normalization
# ---------------------------------------

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def denormalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x * 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5



# ---------------------------------------
#  Utils
# ---------------------------------------

def get_activation(activation):
    if activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()
    elif activation == 'prelu':
        return tf.keras.layers.PReLU(shared_axes=[1, 2, 3])
    else:
        return tf.keras.layers.Activation(activation)
    

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)



# ---------------------------------------
#  Distillation
# ---------------------------------------      
        
def get_margin(std, mean):

    margin = []        
    for (s, m) in zip(std, mean):
        s = abs(s)

        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(-(m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else: 
            margin.append(-3 * s)
    return tf.convert_to_tensor(margin, dtype=tf.float32)