import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

x_train = np.random.rand(5000, 28, 28)
y_train = np.random.randint(0, 10, 5000)
x_test = np.random.rand(1000, 28, 28)
y_test = np.random.randint(0, 10, 1000)

m1 = Sequential([Flatten(input_shape=(28,28)), Dense(256, activation='relu'), Dense(128, activation='relu'), Dense(10, activation='softmax')])
m2 = Sequential([Flatten(input_shape=(28,28)), Dense(256, activation='relu'), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.3), Dense(10, activation='softmax')])

for m in [m1, m2]: m.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

h1 = m1.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=0)
h2 = m2.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=0)

acc1 = m1.evaluate(x_test, y_test, verbose=0)[1]
acc2 = m2.evaluate(x_test, y_test, verbose=0)[1]
print(f"No Dropout: {acc1:.4f} | With Dropout: {acc2:.4f}")
