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

signal = generate_signal(1)      # drone

features = extract_features(signal)

features_q = np.round(features * 127).astype(np.int8)

np.savetxt("input_bird.mem", features_q.astype(np.uint8), fmt="%02x")