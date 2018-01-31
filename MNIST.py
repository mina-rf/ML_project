from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
out_bias = tf.Variable(tf.random_normal([num_classes]))

input = tf.unstack(X, timesteps, 1)

lstm_layer = rnn.BasicLSTMCell(num_hidden, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        # To implement training
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        sess.run(opt, feed_dict={X: batch_x, Y: batch_y})

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images.reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
