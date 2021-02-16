import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, RNN, RepeatVector
from tensorflow.keras.models import Model

# use keras.layers.RNN as loop, following code is equal to:
# x1 = [[2. 2. 2. 2. 2. 2. 2. 2.]
#       [2. 2. 2. 2. 2. 2. 2. 2.]
#       [2. 2. 2. 2. 2. 2. 2. 2.]]
# y = x1
# for _ in range(100):
#    y = y + 1.
#
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

class RNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(RNNCell, self).__init__(**kwargs)

    def call(self, inputs, states):
        prev_output = states[0]
        output = prev_output + 1.
        return output, [output]

batch_size = 3
time_step = 100 # interation
feature_size = 8

x0 = Input((feature_size))
layer0 = RepeatVector(time_step)
tmp = layer0(x0)

x1 = Input((feature_size))

cell = RNNCell(feature_size)
layer1 = RNN(cell)
y = layer1(inputs=tmp, initial_state=x1)

model = Model([x0, x1], y)

print(model.summary())

input0 = np.zeros((batch_size, feature_size), np.float32)
input1 = np.zeros((batch_size, feature_size), np.float32)
input1.fill(2.)
print(input0)
print(input1)

output0 = model.predict([input0, input1])
print(output0)
