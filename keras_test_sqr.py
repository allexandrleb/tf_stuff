import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import itertools
#building a model

model = keras.Sequential()

model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l1(), activation='elu',input_shape=(1,)))
#model.add(keras.layers.Dense(10, activation='softmax'))
#model.add(keras.layers.Dense(64, activation='sigmoid'))
#model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))

#layers.Dense(64, activation='sigmoid')


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['mae'])

#input dataset

xtrain = np.array([np.array([i],dtype='float32') for i in range(1,21,2)])
ytrain = np.array([np.array([math.exp(i)],dtype='float32') for i in xtrain])
xtest = np.array([np.array([i],dtype='float32') for i in range(22,26,2)])
ytest = np.array([np.array([math.exp(i)],dtype='float32') for i in xtest])

dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
dataset = dataset.batch(64).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
val_dataset = val_dataset.batch(64).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

#model.fit(xtrain,ytrain, epochs=10, steps_per_epoch=30,
#          validation_data=test_data
#)


xpred = np.array([np.array([i],dtype = 'float64') for i in range(2,21,2)],dtype = 'float64')

model.evaluate(dataset, steps=30)

predict_results = model.predict(xpred, steps=30)

predictions = list(itertools.islice(predict_results,len(xpred)))

for k,val in enumerate(predictions):
  print(str(xpred[k]) + ' ' + str(val) + ' ' + str(math.exp(xpred[k][0])))
