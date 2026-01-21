import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN #type: ignore
# Generate sine wave dataset
x = np.linspace(0, 100, 500)
data = np.sin(x)
# Prepare training data (windowing)
sequence_length = 10
X, y = [], []
for i in range(len(data) - sequence_length):
	X.append(data[i:i+sequence_length])
	y.append(data[i+sequence_length])
X = np.array(X)
y = np.array(y)
# Reshape to RNN input format (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
# Build RNN model
model = Sequential([
SimpleRNN(32, activation='tanh', input_shape=(sequence_length, 1)),
Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
loss='mse')
# Train model
model.fit(X, y, epochs=20, batch_size=16)
# Predict next values
pred = model.predict(X)
print("Example prediction:")
print(pred[:10])