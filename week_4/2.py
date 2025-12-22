import numpy as np

# XOR dataset - linearly inseparable problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def evaluate(activation_fn, name):
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
    correct = 0
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        a = activation_fn(z)
        pred = 1 if a >= 0.5 else 0
        if pred == y[i]:
            correct += 1
    print(f"\nActivation: {name}")
    print(f"Accuracy: {correct}/{len(X)}")

print("Testing XOR Dataset - Linearly Inseparable Problem")
print("=" * 60)
evaluate(relu, "ReLU")
evaluate(sigmoid, "Sigmoid")
evaluate(tanh, "Tanh")
print("\n" + "=" * 60)
print("Note: XOR is linearly inseparable. Single-layer perceptrons")
print("fail regardless of activation - requires multi-layer network.")
print("=" * 60)
