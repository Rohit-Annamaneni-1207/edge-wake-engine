import numpy as np

class VoiceActivityDetector:
    """
    A lightweight, zero-learn VAD utilizing classical DSP metrics 
    to gate audio streams for edge devices.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=25, stride_duration_ms=10):
        # Design Choice: Standard speech processing windows
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.stride_length = int(sample_rate * (stride_duration_ms / 1000.0))
        
    def _frame_signal(self, signal):
        """Splits the 1D audio signal into overlapping 2D frames."""
        num_frames = 1 + int((len(signal) - self.frame_length) / self.stride_length)
        
        # We pre-allocate the array for memory efficiency
        frames = np.zeros((num_frames, self.frame_length))
        for i in range(num_frames):
            start = i * self.stride_length
            frames[i] = signal[start : start + self.frame_length]
            
        return frames

    def compute_ste(self, frames):
        """
        Computes Short-Time Energy (STE).
        Equation: Sum of squared amplitudes per frame.
        """
        return np.sum(frames ** 2, axis=1)

    def compute_zcr(self, frames):
        """
        Computes Zero-Crossing Rate (ZCR).
        Equation: Rate at which the signal changes algebraic sign.
        """
        # np.sign returns -1, 0, or 1. np.diff finds the changes. 
        # A sign change results in an absolute difference of 2, hence dividing by 2.
        return np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * self.frame_length)

    def is_speech_present(self, signal, energy_threshold, zcr_threshold):
        """
        Evaluates the full signal. Returns True if speech is detected.
        """
        frames = self._frame_signal(signal)
        ste = self.compute_ste(frames)
        zcr = self.compute_zcr(frames)
        
        # Design Choice: Heuristic gating
        # If enough frames exceed the energy threshold AND fall within a vocal ZCR range
        active_frames = (ste > energy_threshold) & (zcr < zcr_threshold)
        
        # If more than 15% of the clip is classified as active, we trigger Stage 1
        return np.mean(active_frames) > 0.15