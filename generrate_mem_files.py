import numpy as np

# Load your quantized weights and biases
q_w1 = np.loadtxt("quantized_w1.txt").astype(np.int8)  # (100, 64)
q_w2 = np.loadtxt("quantized_w2.txt").astype(np.int8)  # (64, 32)
q_w3 = np.loadtxt("quantized_w3.txt").astype(np.int8)  # (32, 4)

q_b1 = np.loadtxt("quantized_b1.txt").astype(np.int8)
q_b2 = np.loadtxt("quantized_b2.txt").astype(np.int8)
q_b3 = np.loadtxt("quantized_b3.txt").astype(np.int8)

def save_mem_transposed(filename, data):
    # CRITICAL FIX: Transpose (.T) before flattening.
    # This ensures N0_W0..9 are lines 0-9 in the file.
    flattened = data.T.flatten().astype(np.uint8)
    np.savetxt(filename, flattened, fmt='%02x')

# Layer 1: 10 batches of (10 inputs x 64 neurons)
for i in range(10):
    batch = q_w1[i*10:(i+1)*10, :] # Shape (10, 64)
    save_mem_transposed(f"q_w1_batch{i}.mem", batch)

# Layer 2: 7 batches (with zero padding for the 7th batch)
for i in range(7):
    if i < 6:
        batch = q_w2[i*10:(i+1)*10, :]
    else:
        # Pad last 6 rows to reach 10 rows
        batch = np.vstack([q_w2[60:64, :], np.zeros((6, 32), dtype=np.int8)])
    save_mem_transposed(f"q_w2_batch{i}.mem", batch)

# Layer 3: 4 batches (32 inputs / 8 per batch in your previous logic, but let's stick to 10)
# To match the RTL logic of "batch_sel * 10", we use 4 batches of 10 rows
for i in range(4):
    start = i * 8 # Using your 8-row step from before
    if start + 8 <= 32:
        batch = np.vstack([q_w3[start:start+8, :], np.zeros((2, 4), dtype=np.int8)])
    save_mem_transposed(f"q_w3_batch{i}.mem", batch)

# Biases
save_mem_transposed("q_b1.mem", q_b1)
save_mem_transposed("q_b2.mem", q_b2)
save_mem_transposed("q_b3.mem", q_b3)

print("Memory files exported with Transposed Neuron-Major alignment.")