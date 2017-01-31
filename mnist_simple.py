""" Uses Tensorflow to identify digits in the MNIST hand-written digit data set """
""" Uses a single layer with a Gradient Descent Optimizer """
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_PIXEL_COUNT = 784

# Pull down the mnist data.
mnist_data = input_data.read_data_sets("mnist_data/", reshape=False, one_hot=True)

# Placeholder for our digit images, which are 28 x 28 pixels (greyscale)
images = tensorflow.placeholder(tensorflow.float32, [None, 28, 28, 1])

# Flatten images
flattened_images = tensorflow.reshape(images, [-1, IMAGE_PIXEL_COUNT])

# Storage for the weights for each pixel, per neuron.
weights = tensorflow.Variable(tensorflow.zeros([IMAGE_PIXEL_COUNT, 10]))

# Bias per neuron.
bias = tensorflow.Variable(tensorflow.zeros([10]))

Answer = tensorflow.nn.softmax(tensorflow.matmul(flattened_images, weights) + bias)

# Correct Answers will be stored here
model_answers = tensorflow.placeholder(tensorflow.float32, [None, 10])

# We use the cross entropy value as our loss function.
cross_entropy = -(tensorflow.reduce_sum(model_answers * tensorflow.log(Answer)))

# Determine if the answer the neural network came up with was correct by comparing against the known answer
is_correct = tensorflow.equal(tensorflow.argmax(Answer, 1), tensorflow.argmax(model_answers, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(is_correct, tensorflow.float32))

init = tensorflow.global_variables_initializer()

optimizer = tensorflow.train.GradientDescentOptimizer(0.004)
training_step = optimizer.minimize(cross_entropy)

session = tensorflow.Session()
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
