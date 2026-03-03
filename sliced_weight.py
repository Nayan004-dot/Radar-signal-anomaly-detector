import numpy as np
#layer one weights
q_w1 = np.loadtxt("quantized_w1.txt")
q_w1 = q_w1.astype(np.int8)

#layer two weights
q_w2 = np.loadtxt("quantized_w2.txt")
q_w2 = q_w2.astype(np.int8)

#layer three weights
q_w3 = np.loadtxt("quantized_w3.txt")
q_w3 = q_w3.astype(np.int8)

for i in range(10):
    batch = q_w1[i*10:(i+1)*10, :].flatten()
    np.savetxt(f"q_w1_batch{i}.mem", batch, fmt='%02x')

for i in range(7):
    batch =  q_w2[i*10:(i+1)*10,:].flatten()
    np.savetxt(f"q_w2_batch{i}.mem",batch,fmt = '%02x')

for i in range(4):
    batch = q_w3[i*10:(i+1)*10,:].flatten()
    np.savetxt(f"q_w3_batch{i}.mem",batch,fmt = '%02x')