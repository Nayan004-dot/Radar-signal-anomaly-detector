import numpy as np

# Load weights
w1 = np.loadtxt("w1.txt")
w2 = np.loadtxt("w2.txt")
w3 = np.loadtxt("w3.txt")
b1 = np.loadtxt("b1.txt")
b2 = np.loadtxt("b2.txt")
b3 = np.loadtxt("b3.txt")

# Calculate Scale Factor
max_val_w1 = np.max(np.abs(w1))
max_val_w2 = np.max(np.abs(w2))
max_val_w3 = np.max(np.abs(w3))
max_val_b1 = np.max(np.abs(b1))
max_val_b2 = np.max(np.abs(b2))
max_val_b3 = np.max(np.abs(b3))

s_w1 = max_val_w1/127
s_w2 = max_val_w2/127
s_w3 = max_val_w3/127
s_b1 = max_val_b1/127
s_b2 = max_val_b2/127
s_b3 = max_val_b3/127

# Quantize and cast to 8-bit signed integers (int8)
# This ensures the values are strictly between -128 and 127
q_w1 = np.round(w1 / s_w1).astype(np.int8)
q_w2 = np.round(w2 / s_w2).astype(np.int8)
q_w3 = np.round(w3 / s_w3).astype(np.int8)
q_b1 = np.round(b1 / s_b1).astype(np.int8)
q_b2 = np.round(b2 / s_b2).astype(np.int8)
q_b3 = np.round(b3 / s_b3).astype(np.int8)

# Save to file
np.savetxt("quantized_w1.txt", q_w1, fmt='%d')
np.savetxt("quantized_w2.txt", q_w2, fmt='%d')
np.savetxt("quantized_w3.txt", q_w3, fmt='%d')
np.savetxt("quantized_b1.txt", q_b1, fmt='%d')
np.savetxt("quantized_b2.txt", q_b2, fmt='%d')
np.savetxt("quantized_b3.txt", q_b3, fmt='%d')
