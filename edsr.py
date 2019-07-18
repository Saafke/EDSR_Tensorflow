from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

def model(x, y, B, F, scale, batch, lr):
    """
    Implementation of EDSR: https://arxiv.org/abs/1707.02921.
    """
    
    #INIT
    scaling_factor = 0.1
    bias_initializer = tf.constant_initializer(value=0.0)
    PS = 3 * (scale*scale) #channels x scale^2
    xavier = tf.contrib.layers.xavier_initializer()
    
    # -- Filters & Biases --
    resFilters = list()
    resBiases = list()

    for i in range(0, B*2):
        resFilters.append( tf.get_variable("resFilter%d" % (i), shape=[3,3,F,F],
           initializer=xavier))
        resBiases.append(tf.get_variable(name="resBias%d" % (i), shape=[F], initializer=bias_initializer ) )

    filter_one = tf.get_variable("resFilter_one", shape=[3,3,3,F], initializer=xavier)
    filter_two = tf.get_variable("resFilter_two", shape=[3,3,F,F], initializer=xavier)
    filter_three = tf.get_variable("resFilter_three", shape=[3,3,F,PS], initializer=xavier)

    bias_one = tf.get_variable(shape=[F], initializer=bias_initializer, name="BiasOne")
    bias_two = tf.get_variable(shape=[F], initializer=bias_initializer, name="BiasTwo")
    bias_three = tf.get_variable(shape=[PS], initializer=bias_initializer, name="BiasThree")

    # -- Model architecture --
    
    # first conv
    x = tf.nn.conv2d(x, filter=filter_one, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_one
    out1 = tf.identity(x)

    # all residual blocks
    for i in range(B):
			x = resBlock(x, (i*2), scaling_factor, resFilters, resBiases)

    # last conv
    x = tf.nn.conv2d(x, filter=filter_two, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_two
    x = x + out1

    # upsample via sub-pixel, equivalent to depth to space
    x = tf.nn.conv2d(x, filter=filter_three, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_three
    out = tf.nn.depth_to_space(x, scale, data_format='NHWC', name="NHWC_output")
    
    # -- -- 

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

def resBlock(inpt, f_nr, scaling_factor, filters, biases):
    x = tf.nn.conv2d(inpt, filter=filters[f_nr], strides=[1, 1, 1, 1], padding='SAME')
    x = x + biases[f_nr]
    x = tf.nn.relu(x)

    x = tf.nn.conv2d(x, filter=filters[f_nr+1], strides=[1, 1, 1, 1], padding='SAME')
    x = x + biases[f_nr+1]
    x = x * scaling_factor

    return inpt + x
