import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

# use tf.while_loop, following code is equal to:
# x = [[2. 2. 2. 2. 2. 2. 2. 2.]
#      [2. 2. 2. 2. 2. 2. 2. 2.]
#      [2. 2. 2. 2. 2. 2. 2. 2.]]
# y = x
# for _ in range(100):
#    y = y + 1.
#
# reference: https://www.tensorflow.org/api_docs/python/tf/while_loop

batch_size = 3
time_step = 100
feature_size = 8

def foo(inputs):

    i = inputs[0]
    x = inputs[1]

    def cond(i, x):
        return i[0][0] < time_step

    def body(i, x):
        return (i + 1, x + 1)

    i, y = tf.while_loop(cond, body, (i, x))
    return y

x0 = Input((1))
x1 = Input((feature_size))
layer = Lambda(foo, name="foo")
y = layer((x0, x1))

model = Model([x0, x1], y)

print(model.summary())

input0 = np.zeros((batch_size, 1), np.int32)
input1 = np.zeros((batch_size, feature_size), np.float32)
input1.fill(2.)

output = model.predict([input0, input1])
print(output)
