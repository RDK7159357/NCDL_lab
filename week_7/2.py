import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore
# Generate sine wave dataset
x = np.linspace(0, 100, 500)
data = np.sin(x)
def prepare_data(seq_len):
	X, y = [], []
	for i in range(len(data) - seq_len):
		X.append(data[i:i + seq_len])
		y.append(data[i + seq_len])
	X = np.array(X).reshape(-1, seq_len, 1)
	y = np.array(y)
	return X, y
def run_experiment(seq_len, hidden_units):
	print(f"\nTesting: sequence_length={seq_len}, hidden_units={hidden_units}")
	X, y = prepare_data(seq_len)
	model = Sequential([
		SimpleRNN(hidden_units, activation='tanh', input_shape=(seq_len, 1)),
		Dense(1)
	])
	model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
	model.fit(X, y, epochs=15, batch_size=16, verbose=0)
	loss = model.evaluate(X, y, verbose=0)
	print(f"Final MSE Loss: {loss:.6f}")
# Try different configurations
run_experiment(5, 16)
run_experiment(10, 32)
run_experiment(20, 64)