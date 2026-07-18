import numpy as np
import librosa

class CustomDSP:
    """
    Mathematical implementations of signal processing transforms.
    """
    
    @staticmethod
    def hamming_window(window_length):
        n = np.arange(window_length)
        return 0.54 - 0.46 * np.cos((2 * np.pi * n) / (window_length - 1))

    @staticmethod
    def compute_dft_matrix(N):
        n = np.arange(N)
        k = n.reshape((N, 1))
        omega = np.exp(-2j * np.pi * k * n / N)
        return omega

    @classmethod
    def compute_stft(cls, frames, n_fft=None):
        num_frames, frame_length = frames.shape
        if n_fft is None:
            n_fft = frame_length
            
        window = cls.hamming_window(frame_length)
        windowed_frames = frames * window 
        
        if n_fft > frame_length:
            padding = np.zeros((num_frames, n_fft - frame_length))
            windowed_frames = np.hstack((windowed_frames, padding))
            
        dft_matrix = cls.compute_dft_matrix(n_fft)
        complex_spectrum = np.dot(windowed_frames, dft_matrix)
        
        power_spectrogram = (np.abs(complex_spectrum) ** 2) / n_fft
        return power_spectrogram[:, :n_fft // 2 + 1]


class MelSpectrogramExtractor:
    """
    Uses custom STFT math, but leverages librosa for biological Mel scaling.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=25, stride_duration_ms=10, n_mels=40):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.stride_length = int(sample_rate * (stride_duration_ms / 1000.0))
        self.n_fft = self.frame_length
        
        # We use librosa JUST to generate the transformation matrix
        # Shape: (n_mels, n_fft // 2 + 1)
        self.mel_filterbank = librosa.filters.mel(
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            n_mels=n_mels
        )
        
    def _frame_signal(self, signal):
        """Splits the 1D audio signal into overlapping 2D frames."""
        num_frames = 1 + int((len(signal) - self.frame_length) / self.stride_length)
        frames = np.zeros((num_frames, self.frame_length))
        for i in range(num_frames):
            start = i * self.stride_length
            frames[i] = signal[start : start + self.frame_length]
        return frames

    def process(self, audio_signal):
        # 1. Frame the raw audio
        frames = self._frame_signal(audio_signal)
        
        # 2. Run our custom mathematical STFT
        power_spectrogram = CustomDSP.compute_stft(frames, self.n_fft)
        
        # 3. Apply the Mel Filterbank via matrix multiplication
        # power_spectrogram: (num_frames, bins)
        # mel_filterbank.T: (bins, n_mels)
        mel_spectrogram = np.dot(power_spectrogram, self.mel_filterbank.T)
        
        # 4. Logarithmic compression for neural network stability
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return log_mel_spectrogram