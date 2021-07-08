import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

tfds.disable_progress_bar()

# we train the model using the train_dataset
# and test it using test_dataset

dataset, metadata = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

# set up training and testing sets

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))


# preprocessing data: changing the range from [0,255] to [0,1]

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# map will normalize each element in the train and test datasets

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# reshape a pic to remove the color dimension

for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))

# plot the image

plt.figure()
plt.imshow(image, cmap=plt.cm.get_cmap("binary"))
plt.colorbar()
plt.grid(False)
plt.show()

# display the 1st 25 pics of the training set + class

# this will ensure the data is correct
# and we're ready to build & train the network

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.get_cmap("binary"))
    plt.xlabel(class_names[label])
plt.show()

# after the first run we load the datasets from cache

train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# setup the layers

# each layer extracts a representation from the data fed to it
# the 1st layer transforms the 2d-array to 1d (28 x 28 to 784)
# 2nd layer weights the input (784 nodes) with 128 neurons fixing parameters during training
# 3rd layer's 10 neurons represents the classes checks each node and outputs a value [0,1]
# it represents the probability of the image belonging to the class
# the sum of all 10 neurons is 1

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# compiling the model

# loss -> measures how far off are we
# optimizer -> adjusts the inner parameters to minimize loss
# metrics -> how we monitor the training and testing

# by setting the optimizer to adam its default value is [0.1]

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# train the model

# .repeat() -> repeat as the epochs limits the training span
# .shuffle(60000) -> randomizes the set so the model won't learn by order
# .batch(32) -> trains using batches of 32 images and labels

BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(
    num_train_examples).batch(BATCH_SIZE).cache().repeat()
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# training data for the model comes from train_dataset
# model learns to associate pics and labels
# epochs=5 sets 5 full iterations of training
# that means: 5 * 60000 = 300000 examples

model.fit(train_dataset, epochs=7, steps_per_epoch=math.ceil(
    num_train_examples / BATCH_SIZE))

# evaluating accuracy

# using the test_dataset to assess accuracy

test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)

# making predictions

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print(predictions.shape)
print(predictions[0])

# prediction is an array of 10 numbers
# describing the confidence of the model that the image belongs to each of the 10 classes of clothes
# .argmax() will show the highest confidence value

np.argmax(predictions[0])

# checking if the prediction is correct

print(test_labels[0])


# graph it out to see the full set of 10 class predictions

def plot_image(j, predictions_array, true_labels, images):
    predictions_array, true_label, im_g = predictions_array[j], true_labels[j], images[j]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(im_g[..., 0], cmap=plt.cm.get_cmap("binary"))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(j, predictions_array, true_label):
    predictions_array, true_label = predictions_array[j], true_label[j]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

plt.show()

# Grab an image from the test dataset

img = test_images[0]
img.shape

# Add the image to a batch where it's the only member

img = np.array([img])
img.shape

# predict the image

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

print(np.argmax(predictions_single[0]))
