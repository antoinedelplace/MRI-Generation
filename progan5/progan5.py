# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Training program that learns the weights of the final version of ProGAN (ProGAN 5)

Requirements
----------
training set : all the png input images must be located in the folder dir_img = "../keras_png_slices_data/data/"

Return
----------
Print information about the loss functions and the training time (useful for post-processing analysis)
Save several files :
- data_train_nblock_{}_transition_{}.npy                           : 4 input images for post-processing visualization
- model.ckpt                                                       : the model weights for post-training image generation
- generated_test_nblock_{n}_transition_{t}_epoch_{e}_batch_{b}.npy : training outputs for post-processing visualization
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
#input_size=256
batch_size=64
z_dim=512
learning_rate=0.001
beta1 = 0.0
epochs = 20
epochs_transition = 20
lambda_adv = 1
lambda_gp = 0.25
eps_drift = 0.001
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

def d_block(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="d_block"):
    with tf.variable_scope(name):
        h1 = conv2d(input_, output_dim, k_h, k_w, d_h, d_w, name='g_h1_{}'.format(name))
        h1 = lrelu(h1)

        h2 = conv2d_downscale2d(h1, min(512, output_dim*2), k_h, k_w, d_h*2, d_w*2, name='g_h2_{}'.format(name))
        h2 = lrelu(h2)
    
        return h2

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

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

# Creating the network architecture
def generator(z, transition, blocks, is_transition):
    with tf.variable_scope("generator") as scope:
        h1 = tf.reshape(z, [-1, 1, 1, z_dim])
        h1 = conv2d(h1, nb_filters[0], 4, 4, 1, 1, padding=[[0, 0], [3, 3], [3, 3], [0, 0]], name='g_h1') # output: 4, 4, 512
        h1 = lrelu(h1)

        h2 = conv2d(h1, nb_filters[0], 3, 3, 1, 1, name='g_h2')
        h2 = lrelu(h2)

        if blocks == 1:
            h3 = conv2d(h2, 1, 1, 1, 1, 1, name='g_h3_{}'.format(blocks)) # output: 4, 4, 1
        else:
            batch_size = tf.shape(h2)[0]

            hx = h2
            for i in range(1, blocks-1):
                hx = g_block(hx, [batch_size, size_block[i], size_block[i], nb_filters[i]], 5, 5, 1, 1, name="g_block{}".format(str(i))) # output: 2**(2+i), 2**(2+i), 2**(10-i)
                
            hx2 = g_block(hx, [batch_size, size_block[blocks-1], size_block[blocks-1], nb_filters[blocks-1]], 5, 5, 1, 1, name="g_block{}".format(str(blocks-1))) # output: 2**(2+blocks-1), 2**(2+blocks-1), 2**(10-(blocks-1))
            h3 = conv2d(hx2, 1, 1, 1, 1, 1, name='g_h3_{}'.format(blocks)) # output: 2**(1+blocks), 2**(1+blocks), 1

            if (is_transition):
                h3old = conv2d(hx, 1, 1, 1, 1, 1, name='g_h3_{}'.format(blocks-1)) # output: 2**blocks, 2**blocks, 1
                h3old = upscale2d(h3old) # output: 2**(1+blocks), 2**(1+blocks), 1
                
                h3 = tf.add_n([h3*transition, h3old*(1-transition)])
            
        h3 = tf.nn.tanh(h3)
        return h3

def discriminator(image, transition, blocks, is_transition, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # input: 2**(1+blocks), 2**(1+blocks), 1
        
        if blocks == 1:
            h1 = conv2d(image, nb_filters[blocks-1], 1, 1, 1, 1, name='d_h1_{}'.format(blocks)) # output: 4, 4, 512
            hx = lrelu(h1)
        else:
            h1 = conv2d(image, nb_filters[blocks-1], 1, 1, 1, 1, name='d_h1_{}'.format(blocks)) # output: 2**(1+blocks), 2**(1+blocks), 2**(11-blocks)
            h1 = lrelu(h1)
            hx2 = d_block(h1, nb_filters[blocks-1], 5, 5, 1, 1, name="d_block{}".format(str(blocks-1))) # output: 2**(2+i), 2**(2+i), 2**(10-i)

            if (is_transition):
                h1old = tf.nn.avg_pool(image, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                h1old = conv2d(h1old, nb_filters[blocks-2], 1, 1, 1, 1, name='d_h1_{}'.format(blocks-1)) # output: 2**blocks, 2**blocks, 2**(12-blocks)
                h1old = lrelu(h1old)

                hx2 = tf.add_n([hx2*transition, h1old*(1-transition)])
        
            hx = hx2 
            for i in range(1, blocks-1):
                hx = d_block(hx, nb_filters[blocks-1-i], 5, 5, 1, 1, name="d_block{}".format(str(blocks-1-i)))

        h2 = minibatch_stddev_layer(hx)
        h2 = conv2d(h2, nb_filters[0], 3, 3, 1, 1, name='d_h2')
        h2 = lrelu(h2)
        
        h3 = conv2d(h2, nb_filters[0], 4, 4, 4, 4, name='d_h3') # output: 1*1*512
        h3 = lrelu(h3)
        
        h4 = tf.reshape(h3, [-1, nb_filters[0]])
        h4 = linear(h4, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

for step in range(0, 13): # 13 steps to reach 256*256 resolution
    nblock = (step+1)//2+1 # from 1 to 7
    is_transition = step%2 # 0 or 1
    print("Beginning nblock={} and is_transition={}".format(nblock, is_transition))
    tf.reset_default_graph()
    
    # Network IO
    img_path = tf.placeholder(tf.string, shape=[])
    transition = tf.placeholder(tf.float32, shape=[])
    real_images = tf.placeholder(tf.float32, [None, 2**(1+nblock), 2**(1+nblock), 1])
    z = tf.placeholder(tf.float32, [None, z_dim])

    # Image preprocessing
    raw_img = tf.read_file(img_path)
    png_img = tf.io.decode_png(raw_img, channels=1)
    png_img = tf.cast(png_img, tf.float32)
    png_img = tf.subtract(png_img, 127.5)
    png_img = tf.divide(png_img, 127.5)
    png_img1 = tf.nn.avg_pool(tf.expand_dims(png_img, 0), [1, 2**(7-nblock), 2**(7-nblock), 1], [1, 2**(7-nblock), 2**(7-nblock), 1], padding='SAME')
    png_img2 = upscale2d(tf.nn.avg_pool(tf.expand_dims(png_img, 0), [1, 2**(8-nblock), 2**(8-nblock), 1], [1, 2**(8-nblock), 2**(8-nblock), 1], padding='SAME'))
    png_img = tf.squeeze(png_img1*transition+png_img2*(1-transition), 0)

    # Models
    G = generator(z, transition, nblock, is_transition)
    D_real, D_real_logits = discriminator(real_images, transition, nblock, is_transition, reuse=False)
    D_fake, D_fake_logits = discriminator(G, transition, nblock, is_transition, reuse=True)

    # Gradient penalty
    alpha = tf.random_uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0., maxval=1.)
    mod_real_images = real_images + alpha*(G - real_images)
    D_mod_real, D_mod_real_logits = discriminator(mod_real_images, transition, nblock, is_transition, reuse=True)

    # Loss functions
    d_loss_real = - tf.reduce_mean(D_real_logits)
    d_loss_fake = tf.reduce_mean(D_fake_logits)

    gradients = tf.gradients(D_mod_real_logits, [mod_real_images])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square((slopes - 1)))

    d_loss = lambda_adv*(d_loss_real+d_loss_fake) + lambda_gp*gradient_penalty + eps_drift*tf.reduce_mean(tf.square(D_real_logits))
    g_loss = - d_loss_fake

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
    if step != 0:
        # Loading weights of previous step
        tensors_to_load = build_tensors_in_checkpoint_file(get_tensors_in_checkpoint_file(file_name="model.ckpt"))
        loader = tf.train.Saver(tensors_to_load)
        loader.restore(sess, "model.ckpt")

    # Saving training images
    if is_transition:
        np.save("data_train_nblock_{}_transition_{}.npy".format(nblock, 0.5), np.array([sess.run(png_img, feed_dict={img_path: data_train[i], transition: 0.5}) for i in range(0, 4)]))
    else:
        np.save("data_train_nblock_{}_transition_{}.npy".format(nblock, 1.0), np.array([sess.run(png_img, feed_dict={img_path: data_train[i], transition: 1.0}) for i in range(0, 4)]))

    # Training
    tps = time.time()
    epochs_progressive = epochs if (nblock == 7 and is_transition == 0) else epochs_transition
    for epoch in range(epochs_progressive):
        np.random.shuffle(data_train)
        for batch in range(len(data_train)//batch_size):
            trans = (epoch*len(data_train)//batch_size+batch)/(epochs_progressive*len(data_train)//batch_size) if is_transition else 1.0
            batch_size_local = len(data_train)%batch_size//4*4 if (batch+1)*batch_size > len(data_train) else batch_size
            batch_z = np.random.normal(0, 1, [batch_size_local, z_dim]).astype(np.float32)
            batch_z = batch_z/np.linalg.norm(batch_z, axis=1)[:, None]
            batch_images = np.array([sess.run(png_img, feed_dict={img_path: d, transition: trans}) for d in data_train[batch*batch_size:batch*batch_size+batch_size_local]])
            
            sess.run(d_optim, feed_dict={real_images: batch_images, z: batch_z, transition: trans})
            for i in range(0, 1):
                sess.run(g_optim, feed_dict={real_images: batch_images, z: batch_z, transition: trans})

            errD_fake = sess.run(d_loss_fake, feed_dict={real_images: batch_images, z: batch_z, transition: trans})
            errD_real = sess.run(d_loss_real, feed_dict={real_images: batch_images, z: batch_z, transition: trans})
            errG = sess.run(g_loss, feed_dict={real_images: batch_images, z: batch_z, transition: trans})

            print("{t} - Epoch {e}, Batch {b}: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=errD_fake+errD_real, g=errG))

            # Plotting and saving weights
            if (batch%50 == 1):
                sample_z = np.random.normal(0, 1, [len(data_test), z_dim]).astype(np.float32)
                sample_z = sample_z/np.linalg.norm(sample_z, axis=1)[:, None]
                sample_images = np.array([sess.run(png_img, feed_dict={img_path: d, transition: trans}) for d in data_test])
                generated_images, d_loss_test, g_loss_test = sess.run([G, d_loss, g_loss], feed_dict={real_images: sample_images, z: sample_z, transition: trans})

                np.save("generated_test_nblock_{n}_transition_{t}_epoch_{e}_batch_{b}.npy".format(n=nblock, t=trans, e=epoch, b=batch), generated_images[:9, :, :, 0])
                print("{t} - Epoch {e}, Batch {b} - Testing: d_loss={d}, g_loss={g}".format(t=time.time()-tps, e=epoch, b=batch, d=d_loss_test, g=g_loss_test))

                save_path = saver.save(sess, "model.ckpt")

    sess.close()
