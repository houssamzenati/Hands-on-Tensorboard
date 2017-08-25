import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#	Define a simple convolutionnal layer

def conv_layer(input, channels_in, channels_out):
	w = tf.get_variable('w_conv', [5,5,channels_in, channels_out], initializer=tf.constant_initializer(0))
	b = tf.get_variable('b_conv', [channels_out], initializer=tf.constant_initializer(0))
	conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding='SAME')
	act = tf.nn.relu(conv + b)
	tf.summary.histogram('weights', w)
	tf.summary.histogram('biaises', b)
	tf.summary.histogram('activations', act)
	return act

#	Define fully connected layer, with relu activation function

def fc_layer(input, channels_in, channels_out):
	w = tf.get_variable('w_fc', [channels_in, channels_out], initializer=tf.constant_initializer(0))
	b = tf.get_variable('b_conv', [channels_out], initializer=tf.constant_initializer(0))
	fc = tf.matmul(input,w) + b
	act = tf.nn.relu(fc)
	tf.summary.histogram('weights', w)
	tf.summary.histogram('biaises', b)
	tf.summary.histogram('activations', act)
	return act

#	Setup placeholders, and reshape the data

x = tf.placeholder(tf.float32, shape=[None,28*28], name='image') #placeholder for the input image
y = tf.placeholder(tf.float32, shape=[None,10], name='labels') #placeholder for the input random vector
x_image = tf.reshape(x, [-1, 28, 28, 1]) #[batch_size,H,W,C] since we do not know the batch size we specify it with -1
tf.summary.image('input_image', x_image, 3)

#	Create the network
with tf.variable_scope('conv1'):
	conv1 = conv_layer(x_image, 1, 32) # channel_in is 1 because grey scale images (3 with RGB)
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2'):
	conv2 = conv_layer(pool1, 32, 64)
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	flattened = tf.reshape(pool2, [-1, 7*7*64]) #[batch_size, 3136]

with tf.variable_scope('fc1'):
	fc1 = fc_layer(flattened, 7*7*64, 1024) # 1024 is the number of neurons in the layer
with tf.variable_scope('fc2'):
	logits = fc_layer(fc1, 1024, 10) #We have 10 logits

#	Compute cross entropy as our loss function

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(
		  tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	tf.summary.scalar('cross_entropy', cross_entropy)
#	Use Adam optimizer

with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#	Compute the accuracy

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)
#	Train the model

#	Initialize all the variables
sess = tf.Session()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/mnist_demo/4')
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())


#	Train for 2000 steps

for i in range(2000):
	batch = mnist.train.next_batch(100)

	if i % 5 ==0:
		s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
		writer.add_summary(s, i)
	#Occasionally report accuracy
	if i % 500 ==0:
		[train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
		print('step %d, training accuracy %g' %(i, train_accuracy))

	#Run the training step
	sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})



