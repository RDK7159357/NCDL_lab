import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def train_with_activation(activation_fn, name):
    np.random.seed(42)
    weights = np.random.randn(2) * 0.01
    bias = np.random.randn() * 0.01
    learning_rate = 0.1
    epochs = 10
    for epoch in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            a = activation_fn(z)
            pred = 1 if a >= 0.5 else 0
            error = y[i] - pred
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
    print(f"\nActivation Function: {name}")
    print("Trained weights:", weights)
    print("Trained bias:", bias)
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        a = activation_fn(z)
        pred = 1 if a >= 0.5 else 0
        print(f"Input: {X[i]} -> Prediction: {pred}")

train_with_activation(relu, "ReLU")
train_with_activation(sigmoid, "Sigmoid")
train_with_activation(tanh, "Tanh")
