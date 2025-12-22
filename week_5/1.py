import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def generate_square():
    img = np.zeros((32,32))
    img[8:24,8:24] = 1
    return img

def generate_circle():
    img = np.zeros((32,32))
    rr, cc = np.ogrid[:32, :32]
    mask = (rr-16)**2 + (cc-16)**2 <= 8**2
    img[mask] = 1
    return img

X, y = [], []
for i in range(200):
    X.append(generate_square())
    y.append(0)
    X.append(generate_circle())
    y.append(1)
X = np.array(X).reshape(-1, 32, 32, 1)
y = np.array(y)

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16)

X_test, y_test = [], []
for i in range(20):
    X_test.append(generate_square())
    y_test.append(0)
    X_test.append(generate_circle())
    y_test.append(1)
X_test = np.array(X_test).reshape(-1, 32, 32, 1)
y_test = np.array(y_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
