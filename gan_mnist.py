import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 64
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
lr = 1e-3
eps = 1e-8

def log(x):
	return tf.log(x+eps)
	
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


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
	
def _initialize_weight(size):
	return tf.Variable(xavier_init(size))

def _initialize_bias(size):
	return tf.Variable(tf.zeros(shape=size))
	
#=========G(z)================

z = tf.placeholder(tf.float32, shape=[None,z_dim])
X = tf.placeholder(tf.float32, shape=[None,X_dim])

G_W1 = _initialize_weight([z_dim, h_dim])
G_b1 = _initialize_bias([h_dim])

G_W2 = _initialize_weight([h_dim,X_dim])
G_b2 = _initialize_bias([X_dim])

G_vars = [G_W1, G_b1, G_W2, G_b2]

def G(z):
	h = tf.nn.relu(tf.matmul(z,G_W1) + G_b1)
	logits = tf.matmul(h,G_W2) + G_b2
	prob = tf.nn.sigmoid(logits)
	return logits, prob
	
#==============D(x)==================

D_W1 = _initialize_weight([X_dim, h_dim])
D_b1 = _initialize_bias([h_dim])

D_W2 = _initialize_weight([h_dim, 1])
D_b2 = _initialize_bias([1])

D_vars = [D_W1, D_b1, D_W2, D_b2]

def D(x):
	h = tf.nn.relu(tf.matmul(x,D_W1) + D_b1)
	logits = tf.matmul(h, D_W2) + D_b2
	prob = tf.nn.sigmoid(logits)
	return prob

#============ training ============

_, prob = G(z)
_, X_samples = G(z)

D_real = D(X)
D_fake = D(prob)

D_loss = -tf.reduce_mean(log(D_real)+log(1-D_fake))
G_loss = -tf.reduce_mean(log(D_fake))

G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = G_vars)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = D_vars)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(tf.trainable_variables())
if not os.path.exists('gan-out/'):
	os.makedirs('gan-out/')
	
i = 0
for it in range(1000000):
	X_mb, _ = mnist.train.next_batch(mb_size)
	z_mb = np.random.uniform(-1,1,[mb_size,z_dim])
	
	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, z: z_mb})
	_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {z: z_mb})
	
	if it % 2000 == 0:
		print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

		samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

		fig = plot(samples)
		plt.savefig('gan-out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
		i += 1
		plt.close(fig)





