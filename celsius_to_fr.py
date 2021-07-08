import numpy as np  # For numerical fast numerical calculations
import matplotlib.pyplot as plt  # For making plots
import tensorflow as tf  # Imports tensorflow
from keras.callbacks import History  # Imports keras
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# setting up features (our input) and labels (our expected outcome)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(
        c, fahrenheit_a[i]))

# setting up our layer and model;
# often they are set up as such:
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = History()
model.fit(celsius_q, fahrenheit_a, epochs=500,
          verbose=False, callbacks=[history])
print("Finished training the model...")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
print(history.history['loss'])

print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))
