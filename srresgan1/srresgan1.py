# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Training program that learns the weights of the final version of SRResGAN (SRResGAN 1)

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

def res_block_d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="res_block_d"):
    with tf.variable_scope(name):
        h1 = conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=stddev, name='d_h1_{}'.format(name))
        h1 = lrelu(h1)

        h2 = conv2d(h1, output_dim, k_h, k_w, d_h, d_w, stddev=stddev, name='d_h2_{}'.format(name))

        h3 = tf.add(input_, h2)
        h3 = lrelu(h3)
    
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

def discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        hx = image
        for i in range(5, 11): # output: 4*4*1024
            hx1 = conv2d(hx, 2**i, 4, 4, 2, 2, name='d_hx1_{}'.format(str(i-5)))
            hx1 = lrelu(hx1)
        
            hx2 = res_block_d(hx1, 2**i, 3, 3, 1, 1, name="d_resblock{}".format(str(2*(i-5))))

            hx = res_block_d(hx2, 2**i, 3, 3, 1, 1, name="d_resblock{}".format(str(2*(i-5)+1)))

        h2 = conv2d(hx, 2048, 3, 3, 2, 2, name='d_h2_conv') # output: 2*2*2048
        h2 = lrelu(h2)
        h2 = tf.reshape(h2, [-1, 2*2*2048])

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

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
alpha = tf.random_uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0., maxval=1.)
mod_real_images = real_images*(1-alpha) + 0.5*alpha*tf.math.reduce_std(real_images)*tf.random_uniform(shape=tf.shape(real_images), minval=0., maxval=1.)
D_mod_real, D_mod_real_logits = discriminator(mod_real_images, reuse=True)

# Loss functions
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real, dtype=tf.float32)*0.9))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

gradients = tf.gradients(D_mod_real_logits, [mod_real_images])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
gradient_penalty = tf.reduce_mean(tf.square((slopes - 1)))

d_loss = lambda_adv*(d_loss_real+d_loss_fake) + lambda_gp*gradient_penalty
g_loss = lambda_adv*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake, dtype=tf.float32)*0.9))

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
        batch_z = np.random.normal(0, 1, [batch_size_local, z_dim]).astype(np.float32)
        batch_z = batch_z/np.linalg.norm(batch_z, axis=1)[:, None]
        batch_images = np.array([sess.run(png_img, feed_dict={img_path: d}) for d in data_train[batch*batch_size:batch*batch_size+batch_size_local]])
        
        sess.run(d_optim, feed_dict={real_images: batch_images, z: batch_z})
        for i in range(0, 2):
            sess.run(g_optim, feed_dict={real_images: batch_images, z: batch_z})

        errD_fake = sess.run(d_loss_fake, feed_dict={real_images: batch_images, z: batch_z})
        errD_real = sess.run(d_loss_real, feed_dict={real_images: batch_images, z: batch_z})
        errG = sess.run(g_loss, feed_dict={real_images: batch_images, z: batch_z})

        print("{t} - Epoch {e}, Batch {b}: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=errD_fake+errD_real, g=errG))

        # Plotting and saving weights
        if (batch%50 == 1):
            sample_z = np.random.normal(0, 1, [len(data_test), z_dim]).astype(np.float32)
            sample_z = sample_z/np.linalg.norm(sample_z, axis=1)[:, None]
            sample_images = np.array([sess.run(png_img, feed_dict={img_path: d}) for d in data_test])
            generated_images, d_loss_test, g_loss_test = sess.run([G, d_loss, g_loss], feed_dict={real_images: sample_images, z: sample_z})

            np.save("generated_test_epoch_{e}_batch_{b}.npy".format(e=epoch, b=batch), generated_images[:9, :, :, 0])
            print("{t} - Epoch {e}, Batch {b} - Testing: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=d_loss_test, g=g_loss_test))

            save_path = saver.save(sess, "model.ckpt")

sess.close()
