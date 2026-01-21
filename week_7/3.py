import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense #type: ignore
# Generate sine wave data
x = np.linspace(0, 100, 500)
data = np.sin(x)
sequence_length = 20
X, y = [], []
for i in range(len(data)-sequence_length):
	X.append(data[i:i+sequence_length])
	y.append(data[i+sequence_length])
X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)
# Build RNN model
model = Sequential([
SimpleRNN(32, activation='tanh', return_sequences=True, input_shape=(sequence_length,
1)),
SimpleRNN(32, activation='tanh'),
Dense(1)
])
optimizer = tf.keras.optimizers.Adam(0.001)
# Custom training to inspect gradients
for epoch in range(5):
	with tf.GradientTape() as tape:
		y_pred = model(X)
		loss = tf.reduce_mean(tf.math.square(y_pred - y))
	gradients = tape.gradient(loss, model.trainable_variables)
	print(f"\nEpoch {epoch+1}")
	for g in gradients:
		print(tf.reduce_mean(tf.abs(g)).numpy())
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))