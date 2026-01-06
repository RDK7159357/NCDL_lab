import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

X_test = np.random.rand(100, 28, 28).astype('float32')
inp = layers.Input(shape=(28,28,1))
x = layers.Conv2D(32, (3,3), activation='relu')(inp)
m1 = models.Model(inp, layers.MaxPooling2D((2,2))(x))
m2 = models.Model(inp, layers.AveragePooling2D((2,2))(x))

sample = X_test[0].reshape(1,28,28,1)
print("Max Pooling:", m1.predict(sample, verbose=0).shape)
print("Avg Pooling:", m2.predict(sample, verbose=0).shape)
