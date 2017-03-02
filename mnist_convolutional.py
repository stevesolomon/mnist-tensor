""" Uses Tensorflow to identify digits in the MNIST hand-written digit data set """
""" Uses multiple fully-connected layers """
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_PIXEL_COUNT = 784

layer_sizes = [6, 12, 24, 200, 10]

weights = [tf.Variable(tf.truncated_normal([6, 6, 1, layer_sizes[0]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([5, 5, layer_sizes[0], layer_sizes[1]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([4, 4, layer_sizes[1], layer_sizes[2]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([7*7*layer_sizes[2], layer_sizes[3]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[3], layer_sizes[4]], stddev=0.2))
          ]

biases = [tf.zeros([layer_sizes[0]]),
          tf.zeros([layer_sizes[1]]),
          tf.zeros([layer_sizes[2]]),
          tf.zeros([layer_sizes[3]]),
          tf.zeros([layer_sizes[4]])
         ]

# We're going to use a variable learning rate so just init as a placeholder.
learning_rate_placeholder = tf.placeholder(tf.float32)

# Placeholder for the probability of keeping a neuron.
keep_probability = tf.placeholder(tf.float32)
TRAINING_KEEP_PROBABILITY = 0.75
TEST_KEEP_PROBABILITY = 1.0

# Pull down the mnist data.
mnist_data = input_data.read_data_sets("mnist_data/", reshape=False, one_hot=True)

# Placeholder for our digit images, which are 28 x 28 pixels (greyscale)
images = tf.placeholder(tf.float32, [None, 28, 28, 1])

layer1 = tf.nn.relu(tf.nn.conv2d(images, weights[0], strides=[1, 1, 1, 1], padding='SAME') + biases[0])
layer2 = tf.nn.relu(tf.nn.conv2d(layer1, weights[1], strides=[1, 2, 2, 1], padding='SAME') + biases[1])
layer3 = tf.nn.relu(tf.nn.conv2d(layer2, weights[2], strides=[1, 2, 2, 1], padding='SAME') + biases[2])

# Layer 4 is a fully connected layer, so reshape the previous layer into a 1D array.
reshaped_layer = tf.reshape(layer3, shape=[-1, 7 * 7 * layer_sizes[2]])

layer4 = tf.nn.relu(tf.matmul(reshaped_layer, weights[3]) + biases[3])
layer4_dropout = tf.nn.dropout(layer4, keep_probability)

logits = tf.matmul(layer4_dropout, weights[4]) + biases[4]
layer5 = tf.nn.softmax(logits)

# Correct Answers will be stored here
model_answers = tf.placeholder(tf.float32, [None, 10])

# We use the cross entropy value as our loss function.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, model_answers)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# Determine if the answer the neural network came up with was correct by comparing against the known answer
is_correct = tf.equal(tf.argmax(layer5, 1), tf.argmax(model_answers, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
training_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

training_results = []
test_results = []

def run_training_step(results, lr):
    # Load in a batch of 100 images at a time, along with the correct answers
    curr_images, currmodel_answers = mnist_data.train.next_batch(100)
    training_data = {images: curr_images, model_answers: currmodel_answers, learning_rate_placeholder: lr, keep_probability: TRAINING_KEEP_PROBABILITY}

    # Run a training session!
    session.run(training_step, feed_dict=training_data)

    results.append(session.run([accuracy, cross_entropy], feed_dict=training_data))

def run_test_step(results, lr):
    print(".", end="", flush=True)
    results.append(session.run([accuracy, cross_entropy],
                               feed_dict={
                                   images: mnist_data.test.images,
                                   model_answers:mnist_data.test.labels,
                                   learning_rate_placeholder: lr,
                                   keep_probability: TEST_KEEP_PROBABILITY
                                   }))

for i in range(10000):

    # Add in learning rate decay 
    min_learning_rate = 0.0001
    max_learning_rate = 0.003
    decay_speed = 2000.0
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    run_training_step(training_results, lr)

    if i % 100 == 0:
        run_test_step(test_results, lr)

run_test_step(test_results, lr)

for i, step in enumerate(training_results):
    print("Training Step #" + str(i) + ": " + str(step))

for i, step in enumerate(test_results):
    print("Test Step #" + str(i) + ": " + str(step))
