import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

# repeat layers with the same name, following code is equal to:
# x = [[2. 2. 2. 2. 2. 2. 2. 2.]
#      [2. 2. 2. 2. 2. 2. 2. 2.]
#      [2. 2. 2. 2. 2. 2. 2. 2.]]
# y = x
# for _ in range(100):
#    y = y + 1.
#
# but it will create a lot of layers

batch_size = 3
time_step = 100 # interation
feature_size = 8

x = Input((feature_size))
y = tf.identity(x)

layer = Lambda(lambda x: x+1, name="my_lambda")
for _ in range(time_step):
  y = layer(y)

model = Model(x, y)

print(model.summary())

input0 = np.zeros((batch_size, feature_size), np.float32)
input0.fill(2.)

print(input0)

output0 = model.predict(input0)
print(output0)
