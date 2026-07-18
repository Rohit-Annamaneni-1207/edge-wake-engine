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

### 3. The Physics and Math of Audio Signals

**Stationarity vs. Pseudo-Stationarity**
*   **Stationary Signals:** A signal is mathematically stationary (Wide-Sense Stationary) if its statistical properties (mean, variance, autocorrelation) remain constant over time. Human speech is highly non-stationary because the sounds we make constantly shift.
*   **The 25ms Window (Pseudo-Stationarity):** The human vocal organs (tongue, jaw, vocal cords) have physical mass and cannot meaningfully change shape faster than 20 to 30 milliseconds. By analyzing audio in 25ms chunks, we isolate a moment where the vocal tract is frozen. Within this micro-window, the sound is "pseudo-stationary," allowing us to apply Fourier Transforms without smearing the frequencies.
*   **Overlapping Strides (10ms):** We slide the 25ms window forward in 10ms steps. This overlapping ensures we do not miss rapid, transient micro-sounds (like the pop of a "P" or "K") that might fall on a frame boundary. It also provides a smooth, high-resolution continuous sequence for the downstream neural networks.

**What Frequencies Are We Capturing?**
While the signal's amplitude oscillates within the 25ms window, the "recipe" of frequencies creating that sound is locked in place. During this window, we extract:
*   **Fundamental Frequency ($f_0$):** The raw pitch of the voice, dictated by how fast the vocal cords are vibrating.
*   **Formants ($F_1, F_2, F_3 \dots$):** The resonant peaks created by the specific physical shape of the throat and mouth acting as an acoustic filter.

**Zero-Crossing Rate (ZCR) Mechanics**
*   **The Math:** ZCR measures how many times the audio wave crosses the $0$ line within a frame. 
    $$ZCR = \frac{1}{2N} \sum_{n=2}^{N} |\text{sgn}(x(n)) - \text{sgn}(x(n-1))|$$
*   **The Application:** Low ZCR corresponds to slow, periodic sounds like vowels (low frequency). High ZCR corresponds to chaotic, rapid vibrations like the "S" sound, wind, or mechanical fan noise (high frequency). It acts as a zero-compute frequency discriminator to block non-human background noise.

**Why Audio Signals Go Negative**
*   **Physics of Sound:** Sound is a mechanical wave. In an audio array, $0$ does not mean "nothingness"; it represents the neutral, ambient atmospheric pressure of a silent room. Positive values represent air compression (higher pressure), and negative values represent air rarefaction (lower pressure/vacuum).
*   **Digital Encoding (PCM):** When a compression wave hits a microphone, the diaphragm pushes inward (positive voltage). When a rarefaction wave hits, it pulls outward (negative voltage). This is converted to digital numbers via 16-bit signed Pulse-Code Modulation (PCM), where silence is $0$, maximum compression is $+32767$, and maximum rarefaction is $-32768$.

### 4. Acoustic Feature Extraction (The Mel Scale)

**Why Raw STFT is Insufficient**
A raw Short-Time Fourier Transform (STFT) spaces frequency bins linearly. However, human hearing is logarithmic. We are highly sensitive to micro-variations at low frequencies (pitch, formants) but biologically deaf to identical variations at high frequencies. Feeding a linear spectrogram into a neural network forces it to allocate unnecessary parameters to model biologically irrelevant high-frequency noise.

**The Mel Filterbank**
To solve this, we apply a Mel Filterbank—a matrix of overlapping triangular filters. 
*   **Low Frequencies:** Filters are narrow and dense, preserving high resolution for critical speech features.
*   **High Frequencies:** Filters are wide and sparse, aggressively averaging and compressing high-frequency bins.
Taking the dot product of the STFT power spectrogram and the Mel Filterbank matrix mathematically warps the acoustic data to map directly to human biological perception.

**Numerical Stability (Decibel Conversion)**
Acoustic power has an extreme dynamic range (e.g., $10^6$ difference between loud vowels and quiet breaths). To prevent exploding gradients during Backpropagation Through Time (BPTT), the Mel-Spectrogram is passed through a logarithmic function ($10 \cdot \log_{10}(S / S_{ref})$), converting the raw power into Decibels (dB). This tightly bounds the feature values, creating a smooth optimization landscape for the neural network.

### 5. The Hybrid Feature Extraction Pipeline

**1. The Hamming Window (Anti-Aliasing)**
Chopping audio into discrete 25ms frames creates artificial, jagged boundaries at the start and end of the frame. The Fourier Transform misinterprets these jagged cuts as high-frequency noise (Spectral Leakage). We apply a Hamming Window—a mathematical curve that forces the edges of the audio frame to smoothly taper to zero, eliminating the artificial noise.

**2. The Digital Prism (DFT Matrix)**
The Discrete Fourier Transform (DFT) acts as a prism, splitting the time-domain audio wave into its constituent frequencies. By pre-computing a matrix of complex sinusoids using Euler's formula ($e^{-j\omega}$), we can transform an entire batch of audio frames into the frequency domain via a single, highly optimized matrix multiplication. 

**3. Biological Translation (Mel Filterbank)**
The raw STFT outputs linear frequencies, but human hearing is logarithmic. We use a Mel Filterbank to warp the linear spectrogram. It applies narrow, high-resolution filters to low frequencies (where humans are highly sensitive) and massive, low-resolution filters to high frequencies (averaging out biologically irrelevant noise).

**4. Decibel Compression (Optimization Stability)**
Raw acoustic power has a mathematically violent dynamic range. Feeding these raw power values into a neural network guarantees exploding gradients. We apply logarithmic compression ($10 \cdot \log_{10}(S / S_{ref})$) to convert the raw power into Decibels (dB). This preserves the acoustic relationships while compressing the values into a stable, narrow numerical range that is safe for gradient descent.