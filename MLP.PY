import numpy as np

# ============================================================
# 1. Signal Simulation
# ============================================================

Fs = 1000
T = 1
N = Fs * T
t = np.linspace(0, T, N, endpoint=False)

def generate_signal(label):

    noise = 0.2 * np.random.randn(N)

    if label == 0:  # Drone
        f = 100
        signal = (np.sin(2*np.pi*f*t) +
                  0.5*np.sin(2*np.pi*2*f*t) +
                  0.3*np.sin(2*np.pi*3*f*t))

    elif label == 1:  # Bird
        f0 = 50
        f1 = 200
        signal = np.sin(2*np.pi*(f0 + (f1-f0)*t)*t)

    elif label == 2:  # Car
        f = 20 + 10*np.sin(2*np.pi*1*t)
        signal = np.sin(2*np.pi*f*t)

    else:  # Noise
        signal = np.random.normal(0, 1, N)

    return signal + noise


# ============================================================
# 2. FFT Feature Extraction
# ============================================================

def extract_features(signal):

    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals[:N//2])

    fft_vals = fft_vals / (np.max(fft_vals) + 1e-9)

    # Take first 50 frequency bins (fixed)
    features = fft_vals[:50]

    return features


# ============================================================
# 3. Create Dataset
# ============================================================

samples_per_class = 200
num_classes = 4

X_list = []
Y_list = []

for label in range(num_classes):
    for _ in range(samples_per_class):

        signal = generate_signal(label)
        features = extract_features(signal)

        X_list.append(features)

        one_hot = np.zeros(num_classes)
        one_hot[label] = 1
        Y_list.append(one_hot)

X = np.array(X_list)
Y = np.array(Y_list)

print("Dataset shape:", X.shape, Y.shape)


# ============================================================
# 4. Shuffle Dataset
# ============================================================

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]


# ============================================================
# 5. Train/Test Split (80/20)
# ============================================================

split = int(0.8 * X.shape[0])

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================================
# 6. Activation Functions
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
# 7. Layers
# ============================================================

def hidden_layer(x, w, b):
    z = np.matmul(x, w) + b
    a = Relu(z)
    return z, a

def output_layer(x, w, b):
    z = np.matmul(x, w) + b
    a = softmax(z)
    return z, a


def forward(x, w1, b1, w2, b2):
    z1, a1 = hidden_layer(x, w1, b1)
    z2, a2 = output_layer(a1, w2, b2)
    return z1, a1, z2, a2


# ============================================================
# 8. Backpropagation
# ============================================================

def training_layer(x, y, z1, A1, z2, A2, w1, b1, w2, b2, lr):

    m = x.shape[0]

    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * relu_derivative(z1)
    dW1 = x.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    w2 -= lr * dW2
    b2 -= lr * db2
    w1 -= lr * dW1
    b1 -= lr * db1

    return w1, b1, w2, b2


# ============================================================
# 9. Initialize Parameters
# ============================================================

np.random.seed(0)

n_x = 50
n_h = 16
n_y = 4

w1 = np.random.randn(n_x, n_h) * 0.01
b1 = np.zeros((1, n_h))
w2 = np.random.randn(n_h, n_y) * 0.01
b2 = np.zeros((1, n_y))


# ============================================================
# 10. Training
# ============================================================

lr = 0.01
epochs = 3000

for i in range(epochs):

    z1, a1, z2, a2 = forward(X_train, w1, b1, w2, b2)

    loss = -np.mean(np.sum(Y_train * np.log(a2 + 1e-9), axis=1))

    w1, b1, w2, b2 = training_layer(
        X_train, Y_train, z1, a1, z2, a2,
        w1, b1, w2, b2,
        lr
    )

    if i % 500 == 0:
        print("Epoch:", i, "Loss:", loss)


# ============================================================
# 11. Train Accuracy
# ============================================================

_, _, _, train_pred = forward(X_train, w1, b1, w2, b2)
train_acc = np.mean(np.argmax(train_pred, axis=1) ==
                    np.argmax(Y_train, axis=1))

print("Train Accuracy:", train_acc)


# ============================================================
# 12. Test Accuracy (REAL PERFORMANCE)
# ============================================================

_, _, _, test_pred = forward(X_test, w1, b1, w2, b2)
test_acc = np.mean(np.argmax(test_pred, axis=1) ==
                   np.argmax(Y_test, axis=1))

print("Test Accuracy:", test_acc)


print("Model saved successfully.")
np.savetxt("w1.txt", w1)
np.savetxt("b1.txt", b1)
np.savetxt("w2.txt", w2)
np.savetxt("b2.txt", b2)