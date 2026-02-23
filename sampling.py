import numpy as np
import matplotlib.pyplot as plt

# generating signal
Fs = 30e3      # sampling frequency
F = 1e3        # 1 kHz
A = 10

T = 1 / F

t = np.linspace(0, 2*T, 100)
x = A * np.sin(2 * np.pi * F * t)

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Original Signal")
plt.grid(True)

# sampling the signal
Ts = 1 / Fs
ts = np.arange(0, 2*T, Ts)
xs = A * np.sin(2 * np.pi * F * ts)

plt.subplot(2, 1, 2)
plt.plot(ts, xs, ".")
plt.title("Sampled Signal")
plt.grid(True)

plt.tight_layout()
plt.show()