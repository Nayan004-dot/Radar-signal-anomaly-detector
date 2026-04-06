# Radar Signal Classification: ML Training & Quantization Pipeline

This repository contains the software baseline and optimization pipeline for a Radar Signal Classifier. It handles everything from synthetic signal generation and feature extraction to training a 3-layer Multi-Layer Perceptron (MLP) and performing **Hardware-Aware Quantization** for FPGA deployment.

## System Overview
The pipeline is designed to bridge the gap between high-level Python training and low-level Verilog RTL.

1.  **Signal Simulation:** Generates synthetic radar signatures for four classes: **Drone**, **Bird**, **Car**, and **Background Noise**.
2.  **Feature Extraction:** Utilizes a 100-point Fast Fourier Transform (FFT) to extract frequency-domain features.
3.  **MLP Training:** A custom NumPy-based 3-layer neural network (100-64-32-4) trained with mini-batch gradient descent.
4.  **Quantization:** Converts 32-bit floating-point weights into **8-bit signed integers** (Q7 format) specifically formatted for Verilog `$readmemh` memory initialization.

## Model Architecture
The model is a fully connected MLP implemented from scratch in NumPy:
* **Input Layer:** 100 Features (FFT Bins)
* **Hidden Layer 1:** 64 Neurons + ReLU Activation
* **Hidden Layer 2:** 32 Neurons + ReLU Activation
* **Output Layer:** 4 Neurons + Softmax (for training) / Argmax (for hardware inference)

## Hardware-Aware Quantization Logic
To ensure the model can run on an FPGA with only integer arithmetic, the following quantization strategy was implemented:

* **Scaling Factor ($2^7$):** Weights are scaled by 128 to utilize the full range of a signed 8-bit integer (-128 to 127).
* **Bias Scaling ($2^{14}$):** To maintain mathematical parity during accumulation, biases are scaled by $SCALE\_W^2$ and stored as 32-bit integers.
* **Neuron-Major Export:** Weights are transposed (`.T.flatten()`) before being converted to Hexadecimal. This ensures that the Verilog FSM can read a single neuron's weights sequentially from memory, minimizing address logic complexity.

## Repository Structure
* **training_pipeline.py**: Main script for signal generation, training, and accuracy evaluation.
* **quantization_script.py**: Loads trained weights and exports formatted `.mem` files for RTL.
* **data/**:
    * `test_vectors.mem`: Quantized test samples (8-bit hex).
    * `w1/w2/w3_neuron_major.mem`: Quantized 8-bit weights.
    * `q_b1/b2/b3.mem`: Quantized 32-bit biases.

## Performance
* **Training Strategy:** Mini-batch training (Batch Size: 64) with Learning Rate Decay.
* **Initialization:** He-Initialization to prevent gradient vanishing in ReLU layers.
* **Evaluation:** Includes a Confusion Matrix and Per-Class Precision/Recall/F1-Score analysis.

## Usage
1. Run `ml.py` to train the model and save floating-point weights (`.txt`).
2. Run `export_mem.py` to generate the `.mem` files.
3. Copy the generated `.mem` files to your **Hardware Accelerator RTL** directory for FPGA synthesis.

---
**Hardware Implementation:** For the Verilog RTL and FPGA deployment files corresponding to this model, see the [Radar-Hardware-Accelerator Repository](https://github.com/Nayan004-dot/radar_signal_classifier_inference).
