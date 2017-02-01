""" Uses Tensorflow to identify digits in the MNIST hand-written digit data set """
""" Uses multiple fully-connected layers """
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_PIXEL_COUNT = 784

layer_sizes = [400, 200, 100, 60, 30, 10]

weights = [tf.Variable(tf.truncated_normal([IMAGE_PIXEL_COUNT, layer_sizes[0]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[0], layer_sizes[1]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[1], layer_sizes[2]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[2], layer_sizes[3]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[3], layer_sizes[4]], stddev=0.2)),
           tf.Variable(tf.truncated_normal([layer_sizes[4], layer_sizes[5]], stddev=0.2))
          ]

biases = [tf.zeros([layer_sizes[0]]),
          tf.zeros([layer_sizes[1]]),
          tf.zeros([layer_sizes[2]]),
          tf.zeros([layer_sizes[3]]),
          tf.zeros([layer_sizes[4]]),
          tf.zeros([layer_sizes[5]])
         ]

# Pull down the mnist data.
mnist_data = input_data.read_data_sets("mnist_data/", reshape=False, one_hot=True)

# Placeholder for our digit images, which are 28 x 28 pixels (greyscale)
images = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Flatten images
flattened_images = tf.reshape(images, [-1, IMAGE_PIXEL_COUNT])

layer1 = tf.nn.sigmoid(tf.matmul(flattened_images, weights[0]) + biases[0])
layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights[1]) + biases[1])
layer3 = tf.nn.sigmoid(tf.matmul(layer2, weights[2]) + biases[2])
layer4 = tf.nn.sigmoid(tf.matmul(layer3, weights[3]) + biases[3])
layer5 = tf.nn.sigmoid(tf.matmul(layer4, weights[4]) + biases[4])
layer6 = tf.nn.softmax(tf.matmul(layer5, weights[5]) + biases[5])

# Correct Answers will be stored here
model_answers = tf.placeholder(tf.float32, [None, 10])

# We use the cross entropy value as our loss function.
cross_entropy = -(tf.reduce_sum(model_answers * tf.log(layer6)))

# Determine if the answer the neural network came up with was correct by comparing against the known answer
is_correct = tf.equal(tf.argmax(layer6, 1), tf.argmax(model_answers, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.004)
training_step = optimizer.minimize(cross_entropy)

session = tf.Session()
session.run(init)

training_results = []
test_results = []

def run_training_step(results):
    # Load in a batch of 100 images at a time, along with the correct answers
    curr_images, currmodel_answers = mnist_data.train.next_batch(100)
    training_data = {images: curr_images, model_answers: currmodel_answers}

    # Run a training session!
    session.run(training_step, feed_dict=training_data)

    results.append(session.run([accuracy, cross_entropy], feed_dict=training_data))

def run_test_step(results):
    print(".", end="", flush=True)
    results.append(session.run([accuracy, cross_entropy],
                               feed_dict={
                                   images: mnist_data.test.images,
                                   model_answers:mnist_data.test.labels
                                   }))

for i in range(10000):
    run_training_step(training_results)

    if i % 100 == 0:
        run_test_step(test_results)

run_test_step(test_results)

for i, step in enumerate(training_results):
    print("Training Step #" + str(i) + ": " + str(step))

for i, step in enumerate(test_results):
    print("Test Step #" + str(i) + ": " + str(step))
