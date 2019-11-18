# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to generate images from the trained model of the final version of SRResGAN (SRResGAN 1)

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
z_dim=256

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
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.initializers.he_normal())
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.initializers.he_normal())
        conv = tf.nn.conv2d(input_, w, strides=[ 1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def res_block_g(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="res_block_g"):
    with tf.variable_scope(name):
        h1 = conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=stddev, name='g_h1_{}'.format(name))
        h1 = tf.nn.relu(batch_norm(name='g_bn1_{}'.format(name))(h1))

        h2 = conv2d(h1, output_dim, k_h, k_w, d_h, d_w, stddev=stddev, name='g_h2_{}'.format(name))
        h2 = batch_norm(name='g_bn2_{}'.format(name))(h2)

        h3 = tf.add(input_, h2)
    
        return h3

def upsampling_block(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="upsampling_block"):
    with tf.variable_scope(name):
        h1 = conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=stddev, name='g_h1_{}'.format(name))

        h2 = tf.nn.depth_to_space(h1, 2, name='g_h2_{}'.format(name))
        
        h3 = tf.nn.relu(batch_norm(name='g_bn1_{}'.format(name))(h2))

        return h3

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

# Creating the network architecture
def generator(z):
    with tf.variable_scope("generator") as scope:
        h1 = linear(z, 64 * 16 * 16, 'g_h1_lin')
        h1 = tf.reshape(h1, [-1, 16, 16, 64])
        h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

        hx = h1
        for i in range(0, 16):
            hx = res_block_g(hx, 64, 3, 3, 1, 1, name="g_resblock{}".format(str(i+1)))
        
        h2 = tf.nn.relu(batch_norm(name='g_bn2')(hx))

        h3 = tf.add(h1, h2)
        
        h4 = upsampling_block(h3, 256, 3, 3, 1, 1, name="g_h4") # output: 32, 32, 64

        h5 = upsampling_block(h4, 256, 3, 3, 1, 1, name="g_h5") # output: 64, 64, 64

        h6 = upsampling_block(h5, 256, 3, 3, 1, 1, name="g_h6") # output: 128, 128, 64

        h7 = upsampling_block(h6, 256, 3, 3, 1, 1, name="g_h7") # output: 256, 256, 64

        h8 = conv2d(h7, 1, 9, 9, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

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
