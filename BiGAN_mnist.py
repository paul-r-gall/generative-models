""" A BiGAN data generation network:
see paper: https://arxiv.org/pdf/1605.09782.pdf
architecture and plotting functions lovingly duplicated from
https://github.com/wiseodd/generative-models
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32

#dimension for latent variable
z_dim = 64
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
# number of nodes in hidden layer
h_dim = 128
lr = 1e-3
beta1= 0.75


#if you don't include this, it returns NaN after ~100k iterations.
def log(x):
	return tf.log(x+1e-8)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

#initialize normals with standard deviation scaled by dimension
def scaled_init(size):
	in_dim = size[0]
	scaled_std = 1./(tf.sqrt(in_dim/2.))
	return tf.random_normal(shape=size,stddev = scaled_std)
	
#------------------------	
z = tf.placeholder(tf.float32, shape = [None, z_dim])
X = tf.placeholder(tf.float32, shape= [None, X_dim])

W_G1 = tf.Variable(scaled_init([z_dim,h_dim]))
b_G1 = tf.Variable(tf.zeros(shape=[h_dim]))

W_G2 = tf.Variable(scaled_init([h_dim,X_dim]))
b_G2 = tf.Variable(tf.zeros(shape=[X_dim]))


#G(z) -- the generator net:
def G(z):
	h = tf.nn.relu(tf.matmul(z, W_G1) + b_G1)
	logits = tf.matmul(h, W_G2) + b_G2
	return logits
	
G_vars = [W_G1, b_G1, W_G2, b_G2]

#-------------------------------

W_E1 = tf.Variable(scaled_init([X_dim, h_dim]))
b_E1 = tf.Variable(tf.zeros(shape=[h_dim]))

W_E2 = tf.Variable(scaled_init([h_dim,z_dim]))
b_E2 = tf.Variable(tf.zeros(shape=[z_dim]))

#E(X) --- the encoder net
def E(X):
	h = tf.nn.relu(tf.matmul(X, W_E1) + b_E1)
	enc = tf.matmul(h, W_E2) + b_E2
	return enc
	
E_vars = [W_E1, b_E1, W_E2, b_E2]
	
#--------------------------------

W_D1 = tf.Variable(scaled_init([X_dim + z_dim, h_dim]))
b_D1 = tf.Variable(tf.zeros(shape=[h_dim]))

W_D2 = tf.Variable(scaled_init([h_dim, 1]))
b_D2 = tf.Variable(tf.zeros(shape = [1]))

#D(data, encoded) -- the discriminator net
def D(dat, enc):
	dat_enc = tf.concat([dat,enc],1)
	h = tf.nn.relu(tf.matmul(dat_enc, W_D1) + b_D1)
	logits = tf.matmul(h, W_D2) + b_D2
	probs = tf.nn.sigmoid(logits)
	return probs

D_vars = [W_D1, W_D2, b_D1, b_D2]

#-----------------------

z_sample = E(X)

fake_X = G(z)

D_real = D(X,E(X))
D_fake = D(G(z), z)

D_loss = -tf.reduce_mean(log(D_real) + log(1-D_fake))

G_loss = -tf.reduce_mean(log(D_fake))

E_loss = tf.reduce_mean(log(D_real))

EG_loss = -tf.reduce_mean(log(D_fake)+log(1-D_real))

#E_solver = tf.train.AdamOptimizer(lr).minimize(E_loss, var_list = E_vars)
#G_solver = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list = G_vars)
D_solver = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list = D_vars)

EG_solver = tf.train.AdamOptimizer(lr).minimize(EG_loss, var_list = E_vars + G_vars)
#load previous values
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state('/tmp/BiGAN_mnist')
if ckpt and ckpt.model_checkpoint_path:
	saver = tf.train.Saver()
	saver.restore(sess, ckpt.model_checkpoint_path)
else: 
	sess.run(tf.global_variables_initializer())
	
if not os.path.exists('BiGAN-out/'):
	os.makedirs('BiGan-out/')

i = 0
for iter in range(1000000):
	X_mb, y_mb = mnist.train.next_batch(mb_size)
	z_mb = np.random.randn(mb_size,z_dim)
	
	#_, E_loss_curr = sess.run([E_solver, E_loss], feed_dict = {X: X_mb})
	#_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {z: z_mb})
	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, z: z_mb})
	_, EG_loss_curr = sess.run([EG_solver, EG_loss], feed_dict = {X: X_mb, z: z_mb})
	
	
	
	if iter % 1000 == 0:
		#print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}'
		#.format(iter, D_loss_curr, G_loss_curr, E_loss_curr))
		
		print('Iter: {}; D_loss: {:.4}; EG_loss: {:.4};'.format(iter, D_loss_curr, EG_loss_curr))

		samples = sess.run(fake_X, feed_dict={z: np.random.randn(16, z_dim)})

		fig = plot(samples)
		plt.savefig('BiGAN-out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
		i += 1
		plt.close(fig)

saver = tf.train.Saver()
saver.save(sess, "/tmp/BiGAN_mnist/model.ckpt")

