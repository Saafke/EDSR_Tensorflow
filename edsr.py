from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

def resBlock(inpt, f_nr, scale, filters, biases):
    x = tf.nn.conv2d(inpt, filter=filters[f_nr], strides=[1, 1, 1, 1], padding='SAME')
    x = x + biases[f_nr]
    x = tf.nn.leaky_relu(x)

    x = tf.nn.conv2d(inpt, filter=filters[f_nr+1], strides=[1, 1, 1, 1], padding='SAME')
    x = x + biases[f_nr+1]
    x = x * scale

    return inpt + x

def model(x, y, B, F, y_shape, scale, batch, lr):
    """
    Implementation of EDSR: https://arxiv.org/abs/1707.02921

    Parameters
    ----------
    x:
        low-res image
    y:
        high-res image
    y_shape:
        shape of output
    scale: int
        super-resolution scale
    batch:
        batch-size
    lr:
        learning rate

    Returns
    ----------
    Model
    """
    print("Scale =", scale)
    
    # -- INFO from paper --
    # RELU: not relu outside of resblocks
    # BATCHSIZE: 16
    # NORMALIZATION: NO
    # EDSR: B = 32, F = 256, scaling-factor = 0.1
    # OPTIMIZER: ADAM
    # LR: starts at 0.0001, halves every 200,000 batches
    
    # Networks for scale 3 and 4, are loaded from pre-trained 2.

    #INIT
    scaling_factor = 0.1
    bias_initializer = tf.constant_initializer(value=0.0)
    PS = 3 * (scale*scale)
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
    # first conv2d layer
    x = tf.nn.conv2d(x, filter=filter_one, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_one
    out1 = tf.identity(x)

    # all residual blocks
    for i in range(B):
			x = resBlock(x, (i*2), scaling_factor, resFilters, resBiases)
    print("Out: ", x.get_shape().as_list())

    # last conv2d layer
    x = tf.nn.conv2d(x, filter=filter_two, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_two
    x = x + out1
    print("Out: ", x.get_shape().as_list())

    # upsample via sub-pixel -- depth to space
    x = tf.nn.conv2d(x, filter=filter_three, strides=[1, 1, 1, 1], padding='SAME')
    x = x + bias_three
    print("Out: ", x.get_shape().as_list())

    out = tf.nn.depth_to_space(x, scale, data_format='NHWC')
    print("Out: ", out.get_shape().as_list())
     
    #out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")

    psnr = tf.image.psnr(out, y, max_val=1.0)
    #loss = tf.losses.mean_squared_error(out, y) #L2
    loss = tf.losses.absolute_difference(out, y) #L1
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return out, loss, train_op, psnr