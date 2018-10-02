import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
#building a model

model = keras.Sequential()

model.add(keras.layers.Dense(10, activation='softmax',input_shape=(16,)))


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['mae'])

data = np.float32(np.random.random((1000, 16)))
labels = np.float32(np.random.random((1000, 10)))

val_data = np.float32(np.random.random((100, 16)))
val_labels = np.float32(np.random.random((100, 10)))


dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(16).repeat()


model.fit(dataset, epochs=10, steps_per_epoch=32)


