import numpy as np

# ============================================================
# Activation Functions
# ============================================================

def Relu(z):
    return np.maximum(z, 0)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu_derivative(z):
    return (z > 0).astype(float)

# ============================================================
# Layers
# ============================================================

def hidden_layer(x, w, b):
    z = np.matmul(x, w) + b
    a = Relu(z)
    return z, a

def output_layer(x, w, b):
    z = np.matmul(x, w) + b
    a = softmax(z)
    return z, a

# ============================================================
# Forward
# ============================================================

def forward(x, w1, b1, w2, b2):
    z1, a1 = hidden_layer(x, w1, b1)
    z2, a2 = output_layer(a1, w2, b2)
    return z1, a1, z2, a2

# ============================================================
# Backpropagation
# ============================================================

def training_layer(x, y, z1, A1, z2, A2, w1, b1, w2, b2, lr):

    m = x.shape[0]

    # Output layer
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer
    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * relu_derivative(z1)
    dW1 = x.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update
    w2 -= lr * dW2
    b2 -= lr * db2
    w1 -= lr * dW1
    b1 -= lr * db1

    return w1, b1, w2, b2

# ============================================================
# Initialize Parameters
# ============================================================

np.random.seed(0)

n_x = 3
n_h = 16
n_y = 3

w1 = np.random.randn(n_x, n_h) * 0.01
b1 = np.zeros((1, n_h))
w2 = np.random.randn(n_h, n_y) * 0.01
b2 = np.zeros((1, n_y))

# ============================================================
# Training
# ============================================================

lr = 0.01
epochs = 3000

for i in range(epochs):

    z1, a1, z2, a2 = forward(X, w1, b1, w2, b2)

    loss = -np.mean(np.sum(Y * np.log(a2 + 1e-9), axis=1))

    w1, b1, w2, b2 = training_layer(
        X, Y, z1, a1, z2, a2,
        w1, b1, w2, b2,
        lr
    )

    if i % 500 == 0:
        print("Epoch:", i, "Loss:", loss)

