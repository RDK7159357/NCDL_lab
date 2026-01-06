import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np

X_train = np.random.rand(5000, 28, 28, 1).astype('float32')
y_train = np.random.randint(0, 10, 5000)
X_test = np.random.rand(1000, 28, 28, 1).astype('float32')
y_test = np.random.randint(0, 10, 1000)

m1 = models.Sequential([layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), layers.Dropout(0.25),
    layers.Conv2D(64, (3,3), activation='relu'), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
    layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dropout(0.5), layers.Dense(10, activation='softmax')])
m2 = models.Sequential([layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D((2,2)), layers.Flatten(),
    layers.Dense(128, activation='relu'), layers.Dense(10, activation='softmax')])

for m in [m1, m2]: m.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

h1 = m1.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
h2 = m2.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

print(f"With Dropout: {m1.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
print(f"Without Dropout: {m2.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
