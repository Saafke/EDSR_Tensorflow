from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

def model(x, y, B, F, scale, batch, lr):
    """
    Implementation of MDSR: https://arxiv.org/abs/1707.02921.

    Information from paper:

        - Head: Pre-process module for each scale: two residual blocks with 5x5 kernels.
        - Tail: Upsampling  module for each scale: like edsr.
        - Baseline model: B=16, F=64
        - Final model: B=80, F=64

    """
    #TODO: per scale

    #INIT
    scaling_factor = 0.1
    
    # -- Model architecture ----------------------------------------------------
    
    # first conv
    x = slim.conv2d(x, F, [3, 3])
    
    # head: two resblocks [SCALE DEPENDENT MODULE]
    out = tf.case({tf.math.equal(scale,2): lambda: doubleHeadResBlock(x, F, scaling_factor, scope="scale2Headblock"), 
                 tf.math.equal(scale,3): lambda: doubleHeadResBlock(x, F, scaling_factor, scope="scale3Headblock")},
                                 default=lambda: doubleHeadResBlock(x, F, scaling_factor, scope="scale4Headblock"), exclusive=True)
    
    # body: B resblocks
    for i in range(B):
		x = bodyResBlock(out, F, scaling_factor)
    
    # second conv
    x = slim.conv2d(x, F, [3, 3])
    x = x + out

    # upsample via sub-pixel - equivalent to depth to space [SCALE DEPENDENT MODULE]
    out = tf.case({tf.math.equal(scale,2): lambda: upsampleBlock(x, (2*2)*3, 2, "_x2"), 
                   tf.math.equal(scale,3): lambda: upsampleBlock(x, (3*3)*3, 3, "_x3")},
                                   default=lambda: upsampleBlock(x, (4*4)*3, 4, "_x4"), exclusive=True)
    
    #out = tf.Print(out, [tf.shape(out)[0], tf.shape(out)[1], tf.shape(out)[2], tf.shape(out)[3]], message="LOL:")
    
    # --------------------------------------------------------------------------

    # some outputs
    out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")
    psnr = tf.image.psnr(out, y, max_val=1.0)
    loss = tf.losses.absolute_difference(out, y) #L1

    # Gradient clipping
    optimizer = tf.train.AdamOptimizer(lr)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    return out, loss, train_op, psnr

def doubleHeadResBlock(inpt, F, scaling_factor, scope):
    x = slim.conv2d(inpt, F, [5, 5], scope=scope+"conv1")
    x = slim.conv2d(x, F, [5, 5], scope=scope+"conv2", activation_fn=None)
    x = x * scaling_factor
    out1 = inpt + x

    x = slim.conv2d(out1, F, [5, 5], scope=scope+"conv3")
    x = slim.conv2d(x, F, [5, 5], scope=scope+"conv4", activation_fn=None)
    x = x * scaling_factor
    return out1 + x

def bodyResBlock(inpt, F, scaling_factor):
    x = slim.conv2d(inpt, F, [3, 3])
    x = slim.conv2d(x, F, [3, 3], activation_fn=None)
    x = x * scaling_factor
    return inpt + x

def upsampleBlock(inpt, PS, scale, scope):
    x = slim.conv2d(inpt, PS, [3, 3], scope="UpsampleConv"+scope, activation_fn=None)
    x = tf.nn.depth_to_space(x, scale, data_format='NHWC', name="NHWC_output"+scope)
    return x