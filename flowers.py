import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
# This is a simple keras library import for CNN

import os
import numpy as np
import glob
import shutil
from matplotlib import pylab as plt

# downloading the datasets

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_dir = tf.keras.utils.get_file(
    origin=_URL,
    fname="flow_photos.tgz",
    extract=True
)
base_dir = os.path.join(
    os.path.dirname(zip_dir), 'flower_photos'
)

# creating labels for the classes

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# print total of flowers for each type
# create folder for and move items to train and validation directories

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    train, val = images[:round(len(images)*0.8)
                        ], images[round(len(images)*0.8):]

    # creating the folders

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))

            # moving the files

            shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))

# setup path for sets

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

total_train = len(os.listdir(train_dir))
total_val = len(os.listdir(val_dir))

# data augmentation

BATCH_SIZE = 100
IMG_SHAPE = 150

# flipping picture horizontally

img_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)


# testing the augmentation on a small plot (5)

def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# setting random rotation

img_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    shuffle=True,
    directory=train_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# setting random zoom

img_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    shuffle=True,
    directory=train_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# setting it all together now

img_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.5,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15)
train_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# setting up validation set

img_gen_val = ImageDataGenerator(
    rescale=1./255
)
val_data_gen = img_gen_val.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

# creating the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation='relu',
        input_shape=(150, 150, 3)
    ),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        512,
        activation='relu'
    ),
    tf.keras.layers.Dense(
        5,
        activation=tf.nn.softmax
    )
])

# compiling the model

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# training the model

print("Total val:", total_val)
print("Total training:", total_train)

EPOCHS = 80
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)


# retrieving plot data

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# building plot

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
