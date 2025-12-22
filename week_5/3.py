import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

filters, biases = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
for i in range(8):
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.title(f"Filter {i}")
    plt.axis('off')
    plt.show()

activation_model = models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers[:2]])
activations = activation_model.predict(np.expand_dims(X_test[7], axis=0))
for i in range(8):
    plt.imshow(activations[0][0, :, :, i], cmap='viridis')
    plt.title(f"Feature map {i}")
    plt.axis('off')
    plt.show()
