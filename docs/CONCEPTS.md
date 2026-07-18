# End-to-End Wake Word Detection Concepts

## 1. Cascade Architecture (System-Level Design)
In production ML, deploying a heavy deep learning model for a continuous, always-on task (like listening for a wake word) is computationally prohibitive. It drains batteries, violates thermal constraints on edge devices, and wastes processing power on silence.

To solve this, we use a **Cascade Architecture**:
*   **Stage 0 (The Gate):** A zero-compute classical DSP filter that drops silent or unvoiced audio frames before they ever reach a neural network.
*   **Stage 1 (The Trigger):** A highly compressed, low-power model (e.g., a tiny CNN) that continuously processes incoming data. It is explicitly tuned for **High Recall** to ensure it never misses a true wake word, accepting some False Positives.
*   **Stage 2 (The Verifier):** A computationally heavier model with temporal context (e.g., an LSTM or Transformer). It remains dormant until Stage 1 flags a potential positive. Once triggered, Stage 2 analyzes the buffered audio sequence with **High Precision** to filter out the false positives.

**The Engineering Trade-off:** System latency and power consumption are dictated by the confidence threshold of Stage 1. A lower threshold catches all wake words but forces the expensive Stage 2 model to execute far too frequently.

## 2. Digital Signal Processing (DSP) Pipeline
To optimize for edge deployment and demonstrate rigorous signal handling, this pipeline incorporates classical DSP techniques prior to deep learning inference:

*   **Voice Activity Detection (VAD):** Utilizes **Zero-Crossing Rate (ZCR)** and **Short-Time Energy (STE)**. ZCR measures the rate at which the signal changes sign (high for unvoiced speech/fricatives), while STE measures amplitude. Together, they form a robust, computationally free Stage 0 gate.
*   **Wiener Filtering:** An optimal linear filter applied to the raw waveform to suppress stationary background noise. It computes the local signal-to-noise ratio and attenuates frequency bins dominated by noise, minimizing the mean square error between the estimated and true signals:
    $H(\omega)=\frac{P_s(\omega)}{P_s(\omega)+P_n(\omega)}$
*   **Algorithmic STFT:** The transformation from the 1D time domain to the 2D time-frequency domain is handled via a custom Discrete Fourier Transform (DFT) implementation, utilizing Hamming windows to mitigate spectral leakage at the frame boundaries:
    $X_m(\omega)=\sum_{n=-\infty}^{\infty}x(n)w(n-mR)e^{-j\omega n}$