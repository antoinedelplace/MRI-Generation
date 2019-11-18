# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to generate images from the trained model of the final version of ProGAN (ProGAN 5)

Requirements
----------
model.ckpt : file containing the model weights (output of the training program)
folder "generated_images" in which the images are generated

Return
----------
Generate n_img = 11328 png images from random samples in the folder "generated_images"
Generate the image from the zero sample (called "mean" image)
"""

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import imageio

# Number of images to generate
n_img = 11328

# Hyperparameters
input_size=256
batch_size=64
z_dim=512
size_block = [4, 8, 16, 32, 64, 128, 256]
nb_filters = [512, 512, 256, 128, 64, 32, 16]

# Functions to load weights
def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist=[]
    var_value =[]
    reader = tf.train.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    for i, tensor_name in enumerate(loaded_tensors[0]):
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            if np.shape(tensor_aux) == np.shape(loaded_tensors[1][i]):
                full_var_list.append(tensor_aux)
            else:
                print(tensor_name+' has wrong shape: '+str(np.shape(loaded_tensors[1][i]))+' instead of '+str(np.shape(tensor_aux)))
        except:
            print('Not found: '+tensor_name)
    return full_var_list

# Defining all components of the neural network
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = tf.shape(x)
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x

def get_weight(shape, gain=np.sqrt(2), fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init

    wscale = tf.constant(std, dtype=tf.float32, name='wscale')
    return tf.get_variable('w', shape=shape, initializer=tf.initializers.random_normal()) * wscale

def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        w = get_weight([shape[1], output_size])
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        
        if with_w:
            return tf.matmul(input_, w) + bias, w, bias
        else:
            return tf.matmul(input_, w) + bias
        
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        w = get_weight([k_h, k_w, input_.get_shape().as_list()[-1], output_dim])
        conv = tf.nn.conv2d(input_, w, strides=[ 1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def upscale2d_conv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="upscale2d_conv2d"):
    with tf.variable_scope(name):
        w = get_weight([k_h, k_w, output_shape[-1], input_.get_shape().as_list()[-1]])

        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        return deconv

def conv2d_downscale2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', name="conv2d_downscale2d"):
    with tf.variable_scope(name):
        w = get_weight([k_h, k_w, input_.get_shape().as_list()[-1], output_dim])

        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])*0.25
        
        conv = tf.nn.conv2d(input_, w, strides=[ 1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def g_block(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, name="g_block"):
    with tf.variable_scope(name):
        h1 = upscale2d_conv2d(input_, output_shape, k_h, k_w, d_h*2, d_w*2, name='g_h1_{}'.format(name))
        h1 = lrelu(h1)

        h2 = conv2d(h1, output_shape[-1], k_h, k_w, d_h, d_w, name='g_h2_{}'.format(name))
        h2 = lrelu(h2)
    
        return h2

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

# Creating the network architecture
def generator(z):
    with tf.variable_scope("generator") as scope:
        h1 = tf.reshape(z, [-1, 1, 1, z_dim])
        h1 = conv2d(h1, nb_filters[0], 4, 4, 1, 1, padding=[[0, 0], [3, 3], [3, 3], [0, 0]], name='g_h1') # output: 4, 4, 512
        h1 = lrelu(h1)

        h2 = conv2d(h1, nb_filters[0], 3, 3, 1, 1, name='g_h2')
        h2 = lrelu(h2)

        batch_size = tf.shape(h2)[0]

        hx = h2
        for i in range(1, 7-1):
            hx = g_block(hx, [batch_size, size_block[i], size_block[i], nb_filters[i]], 5, 5, 1, 1, name="g_block{}".format(str(i))) # output: 2**(2+i), 2**(2+i), 2**(10-i)
            
        hx2 = g_block(hx, [batch_size, size_block[7-1], size_block[7-1], nb_filters[7-1]], 5, 5, 1, 1, name="g_block{}".format(str(7-1))) # output: 2**(2+7-1), 2**(2+7-1), 2**(10-(7-1))
        h3 = conv2d(hx2, 1, 1, 1, 1, 1, name='g_h3_{}'.format(7)) # output: 2**(1+7), 2**(1+7), 1
            
        h3 = tf.nn.tanh(h3)
        return h3

# Network IO
z = tf.placeholder(tf.float32, [None, z_dim])

# Generator model
G = generator(z)

# Initialization
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Loading weights
tensors_to_load = build_tensors_in_checkpoint_file(get_tensors_in_checkpoint_file(file_name="model.ckpt"))
loader = tf.train.Saver(tensors_to_load)
loader.restore(sess, "model.ckpt")

# Generating images
def generate_mean():
    print("generate_mean")
    generated_images = sess.run(G, feed_dict={z: np.zeros((1, z_dim))})
    print("generate_mean: Done")
    
    generated_images *= 127.5
    generated_images = (generated_images+127.5).astype("uint8")
    
    imageio.imwrite("generated_images/gimg_mean.png", generated_images[0])

def generate_random(nb_img_batch, start_number):
    print("generate_random: {} / {} - {} %".format(start_number, n_img, int(float(start_number)/n_img*100)))
    sample_z = np.random.normal(0, 1, [nb_img_batch, z_dim]).astype(np.float32)
    sample_z = sample_z/np.linalg.norm(sample_z, axis=1)[:, None]
    generated_images = sess.run(G, feed_dict={z: sample_z})
    
    generated_images *= 127.5
    generated_images = (generated_images+127.5).astype("uint8")
    
    for j in range(0, nb_img_batch):
        imageio.imwrite("generated_images/gimg_{}.png".format(start_number+j), generated_images[j])

generate_mean()

q = n_img//batch_size
r = n_img%batch_size
for i in range(0, q):
    generate_random(batch_size, i*batch_size)
generate_random(r, q*batch_size)

sess.close()
