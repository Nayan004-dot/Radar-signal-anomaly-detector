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
        signal = (np.sin(2 * np.pi * f * t) +
                  0.5 * np.sin(2 * np.pi * 2 * f * t) +
                  0.3 * np.sin(2 * np.pi * 3 * f * t))

    elif label == 1:  # Bird — chirp up to 200 Hz, now captured with 100 bins
        f0 = 50
        f1 = 200
        signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * t)

    elif label == 2:  # Car — low frequency with modulation
        f = 20 + 10 * np.sin(2 * np.pi * 1 * t)
        signal = np.sin(2 * np.pi * f * t)

    else:  # Background noise
        signal = np.random.normal(0, 1, N)

    return signal + noise


# ============================================================
# 2. FFT Feature Extraction  (100 bins → captures up to 100 Hz)
# ============================================================

N_FEATURES = 100   # was 50; doubled to capture bird chirp up to ~200 Hz

def extract_features(signal):
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals[:N // 2])
    fft_vals = fft_vals / (np.max(fft_vals) + 1e-9)
    return fft_vals[:N_FEATURES]


# ============================================================
# 3. Create Dataset
# ============================================================

samples_per_class = 200
num_classes = 4

X_list, Y_list = [], []

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

print(f"Dataset shape: X={X.shape}, Y={Y.shape}")


# ============================================================
# 4. Shuffle & Train/Test Split (80/20)
# ============================================================

np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

split = int(0.8 * X.shape[0])
X_train, Y_train = X[:split], Y[:split]
X_test,  Y_test  = X[split:], Y[split:]

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")


# ============================================================
# 5. Activation Functions
# ============================================================

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ============================================================
# 6. Forward Pass  — 3 layers (2 hidden + 1 output)
# ============================================================

def forward(x, params):
    w1, b1, w2, b2, w3, b3 = (params[k] for k in ['w1','b1','w2','b2','w3','b3'])

    z1 = x @ w1 + b1;   a1 = relu(z1)
    z2 = a1 @ w2 + b2;  a2 = relu(z2)
    z3 = a2 @ w3 + b3;  a3 = softmax(z3)

    cache = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3)
    return a3, cache


# ============================================================
# 7. Backpropagation
# ============================================================

def backward(y, cache, params, lr):
    m = y.shape[0]
    x, z1, a1, z2, a2, a3 = cache['x'], cache['z1'], cache['a1'], \
                              cache['z2'], cache['a2'], cache['a3']
    w1, w2, w3 = params['w1'], params['w2'], params['w3']

    # Output layer (softmax + cross-entropy combined gradient)
    dZ3 = a3 - y
    dW3 = a2.T @ dZ3 / m
    db3 = dZ3.mean(axis=0, keepdims=True)

    # Hidden layer 2
    dA2 = dZ3 @ w3.T
    dZ2 = dA2 * relu_deriv(z2)
    dW2 = a1.T @ dZ2 / m
    db2 = dZ2.mean(axis=0, keepdims=True)

    # Hidden layer 1
    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * relu_deriv(z1)
    dW1 = x.T @ dZ1 / m
    db1 = dZ1.mean(axis=0, keepdims=True)

    params['w3'] -= lr * dW3;  params['b3'] -= lr * db3
    params['w2'] -= lr * dW2;  params['b2'] -= lr * db2
    params['w1'] -= lr * dW1;  params['b1'] -= lr * db1

    return params


# ============================================================
# 8. Cross-Entropy Loss
# ============================================================

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


# ============================================================
# 9. Initialize Parameters  (He initialization for ReLU)
# ============================================================

np.random.seed(0)

n_x  = N_FEATURES   # 100
n_h1 = 64           # first hidden layer
n_h2 = 32           # second hidden layer  ← new
n_y  = num_classes  # 4

params = {
    'w1': np.random.randn(n_x,  n_h1) * np.sqrt(2.0 / n_x),
    'b1': np.zeros((1, n_h1)),
    'w2': np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1),
    'b2': np.zeros((1, n_h2)),
    'w3': np.random.randn(n_h2, n_y)  * np.sqrt(2.0 / n_h2),
    'b3': np.zeros((1, n_y)),
}


# ============================================================
# 10. Mini-Batch Training with LR Decay
# ============================================================

EPOCHS     = 3000
LR_INIT    = 0.1
BATCH_SIZE = 64
DECAY_RATE = 0.99
DECAY_STEP = 100       # decay lr every N epochs

history = {'loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):

    # Learning-rate schedule
    lr = LR_INIT * (DECAY_RATE ** (epoch // DECAY_STEP))

    # Shuffle training data each epoch
    perm = np.random.permutation(X_train.shape[0])
    X_shuf, Y_shuf = X_train[perm], Y_train[perm]

    # Mini-batch loop
    epoch_loss = 0.0
    n_batches  = 0
    for start in range(0, X_shuf.shape[0], BATCH_SIZE):
        Xb = X_shuf[start:start + BATCH_SIZE]
        Yb = Y_shuf[start:start + BATCH_SIZE]

        a3, cache = forward(Xb, params)
        loss_b    = cross_entropy(Yb, a3)
        params    = backward(Yb, cache, params, lr)

        epoch_loss += loss_b
        n_batches  += 1

    avg_loss = epoch_loss / n_batches

    # Validation metrics (full test set, no grad)
    val_pred, _ = forward(X_test, params)
    val_loss    = cross_entropy(Y_test, val_pred)
    val_acc     = np.mean(np.argmax(val_pred, axis=1) == np.argmax(Y_test, axis=1))

    history['loss'].append(avg_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | LR: {lr:.5f} | Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# ============================================================
# 11. Final Accuracy
# ============================================================

train_pred, _ = forward(X_train, params)
train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(Y_train, axis=1))

test_pred, _  = forward(X_test, params)
test_acc  = np.mean(np.argmax(test_pred, axis=1) == np.argmax(Y_test, axis=1))

print(f"\nTrain Accuracy : {train_acc:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")


# ============================================================
# 12. Confusion Matrix
# ============================================================

CLASS_NAMES = ['Drone', 'Bird', 'Car', 'Noise']

y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(test_pred, axis=1)

conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    conf_matrix[t][p] += 1

print("\nConfusion Matrix (rows=True, cols=Predicted):")
header = f"{'':>8}" + "".join(f"{n:>8}" for n in CLASS_NAMES)
print(header)
for i, row in enumerate(conf_matrix):
    print(f"{CLASS_NAMES[i]:>8}" + "".join(f"{v:>8}" for v in row))

# Per-class precision & recall
print("\nPer-Class Metrics:")
print(f"{'Class':>8} {'Precision':>12} {'Recall':>10} {'F1':>8}")
for i, name in enumerate(CLASS_NAMES):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"{name:>8} {precision:>12.4f} {recall:>10.4f} {f1:>8.4f}")


# ============================================================
# 13. Save Weights
# ============================================================

for key, val in params.items():
    np.savetxt(f"{key}.txt", val)

print("\nAll weights saved successfully.")