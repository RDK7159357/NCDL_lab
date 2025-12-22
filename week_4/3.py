import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def d_relu(x):
    return 1 if x > 0 else 0

def investigate_gradients(activation_fn, derivative_fn, name):
    np.random.seed(42)
    weights = np.random.randn(2) * 0.01
    bias = np.random.randn() * 0.01
    learning_rate = 0.1
    epochs = 10
    print(f"\nActivation: {name}")
    for epoch in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            a = activation_fn(z)
            pred = 1 if a >= 0.5 else 0
            error = y[i] - pred
            grad = derivative_fn(z)
            print(f"Epoch {epoch}, Input {X[i]}, Gradient: {grad:.6f}")
            weights += learning_rate * error * X[i] * grad
            bias += learning_rate * error * grad

investigate_gradients(sigmoid, d_sigmoid, "Sigmoid")
investigate_gradients(tanh, d_tanh, "Tanh")
investigate_gradients(relu, d_relu, "ReLU")
