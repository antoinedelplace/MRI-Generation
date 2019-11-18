# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Training program that learns the weights of the final version of DCGAN (DCGAN 3)

Requirements
----------
training set : all the png input images must be located in the folder dir_img = "../keras_png_slices_data/data/"

Return
----------
Print information about the loss functions and the training time (useful for post-processing analysis)
Save several files :
- data_train.npy                         : 4 input images for post-processing visualization
- model.ckpt                             : the model weights for post-training image generation
- generated_test_epoch_{e}_batch_{b}.npy : training outputs for post-processing visualization
"""

import numpy as np
import tensorflow as tf
import os
import time

# Hardware configuration (checking GPU)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Tensorflow version: ", tf.__version__)
print("Test GPU: ", tf.test.gpu_device_name())

# Import training set
dir_img = "../keras_png_slices_data/data/"
data = [os.path.join(dir_img, f) for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f))]
np.random.shuffle(data)
data_train = data[16:]
data_test = data[:16]
print(data_test)

# Hyperparameters
input_size=256
batch_size=64
z_dim=256
learning_rate=0.0002
beta1 = 0.5
epochs = 20
lambda_adv = 1
lambda_gp = 0.25

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
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[ 1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def minibatch_stddev_layer(x, group_size=4, name="MinibatchStddev"):
    with tf.variable_scope(name):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           
        y = tf.reduce_mean(tf.square(y), axis=0)                
        y = tf.sqrt(y + 1e-8)                                   
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      
        y = tf.tile(y, [group_size, s[1], s[2], 1])             
        return tf.concat([x, y], axis=3)

# Creating the network architecture
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
d_bn5 = batch_norm(name='d_bn5')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')
g_bn8 = batch_norm(name='g_bn8')
g_bn9 = batch_norm(name='g_bn9')

def generator(z):
    with tf.variable_scope("generator") as scope:
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [-1, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        
        batch_size = tf.shape(h1)[0]

        h2 = deconv2d(h1, [batch_size, 16, 16, 256], 5, 5, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256], 5, 5, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256], 5, 5, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256], 5, 5, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 256], 5, 5, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 64, 64, 256], 5, 5, 1, 1, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))

        h8 = deconv2d(h7, [batch_size, 128, 128, 128], 5, 5, 2, 2, name='g_h8')
        h8 = tf.nn.relu(g_bn8(h8))

        h9 = deconv2d(h8, [batch_size, 256, 256, 64], 5, 5, 2, 2, name='g_h9')
        h9 = tf.nn.relu(g_bn9(h9))

        h10 = deconv2d(h9, [batch_size, 256, 256, 1], 5, 5, 1, 1, name='g_h10')
        h10 = tf.nn.tanh(h10)

        return h10

def discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        h1 = conv2d(image, 64, 5, 5, 2, 2, name='d_h1_conv') # output: 128*128*64
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, 5, 5, 2, 2, name='d_h2_conv') # output: 64*64*128
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, 5, 5, 2, 2, name='d_h3_conv') # output: 32*32*256
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, 5, 5, 2, 2, name='d_h4_conv') # output: 16*16*512
        h4 = lrelu(d_bn4(h4))

        h5 = conv2d(h4, 1024, 5, 5, 2, 2, name='d_h5_conv') # output: 8*8*1024
        h5 = lrelu(d_bn5(h5))
        h5 = minibatch_stddev_layer(h5) # output: 8*8*1025
        h5 = tf.reshape(h5, [-1, 8*8*1025])

        h6 = linear(h5, 1025, 'd_h6_lin')
        h6 = lrelu(h6)

        h7 = linear(h6, 1, 'd_h7_lin')

        return tf.nn.sigmoid(h7), h7

# Network IO
img_path = tf.placeholder(tf.string, shape=[])
real_images = tf.placeholder(tf.float32, [None, input_size, input_size, 1])
z = tf.placeholder(tf.float32, [None, z_dim])

# Image preprocessing
raw_img = tf.read_file(img_path)
png_img = tf.io.decode_png(raw_img, channels=1)
png_img = tf.cast(png_img, tf.float32)
png_img = tf.subtract(png_img, 127.5)
png_img = tf.divide(png_img, 127.5)

# Models
G = generator(z)
D_real, D_real_logits = discriminator(real_images, reuse=False)
D_fake, D_fake_logits = discriminator(G, reuse=True)

# Gradient penalty
alpha = tf.random_uniform(shape=tf.shape(real_images), minval=0., maxval=1.)
mod_real_images = real_images*(1-alpha) + 0.5*alpha*tf.math.reduce_std(real_images)*tf.random_uniform(shape=tf.shape(real_images), minval=0., maxval=1.)
D_mod_real, D_mod_real_logits = discriminator(mod_real_images, reuse=True)

# Loss functions
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real, dtype=tf.float32)*0.9))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

gradients = tf.gradients(D_mod_real_logits, [mod_real_images])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean(tf.square((slopes - 1)))

d_loss = lambda_adv*(d_loss_real+d_loss_fake) + lambda_gp*gradient_penalty
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake, dtype=tf.float32)*0.9))

# Optimizers
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
        
d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

# Initialization
init = tf.global_variables_initializer()

saver = tf.train.Saver(var_list=t_vars)

sess = tf.Session()
sess.run(init)

# Saving training images
np.save("data_train.npy", np.array([sess.run(png_img, feed_dict={img_path: data_train[i]}) for i in range(0, 4)]))

# Training
tps = time.time()
for epoch in range(epochs):
    np.random.shuffle(data_train)
    for batch in range(len(data_train)//batch_size):
        batch_size_local = len(data_train)%batch_size if (batch+1)*batch_size > len(data_train) else batch_size
        batch_z = np.random.uniform(-1, 1, [batch_size_local, z_dim]).astype(np.float32)
        batch_images = np.array([sess.run(png_img, feed_dict={img_path: d}) for d in data_train[batch*batch_size:batch*batch_size+batch_size_local]])
        
        sess.run(d_optim, feed_dict={real_images: batch_images, z: batch_z})
        for i in range(0, 3):
            sess.run(g_optim, feed_dict={real_images: batch_images, z: batch_z})

        errD_fake = sess.run(d_loss_fake, feed_dict={real_images: batch_images, z: batch_z})
        errD_real = sess.run(d_loss_real, feed_dict={real_images: batch_images, z: batch_z})
        errG = sess.run(g_loss, feed_dict={real_images: batch_images, z: batch_z})

        print("{t} - Epoch {e}, Batch {b}: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=errD_fake+errD_real, g=errG))

        # Plotting and saving weights
        if (batch%50 == 1):
            sample_z = np.random.uniform(-1, 1, [len(data_test), z_dim]).astype(np.float32)
            sample_images = np.array([sess.run(png_img, feed_dict={img_path: d}) for d in data_test])
            generated_images, d_loss_test, g_loss_test = sess.run([G, d_loss, g_loss], feed_dict={real_images: sample_images, z: sample_z})

            np.save("generated_test_epoch_{e}_batch_{b}.npy".format(e=epoch, b=batch), generated_images[:9, :, :, 0])
            print("{t} - Epoch {e}, Batch {b} - Testing: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=d_loss_test, g=g_loss_test))

            save_path = saver.save(sess, "model.ckpt")

sess.close()
