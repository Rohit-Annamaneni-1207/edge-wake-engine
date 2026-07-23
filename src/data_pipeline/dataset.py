# import torch
# import librosa
# import numpy as np
# from torch.utils.data import Dataset
# from .dsp_utils import MelSpectrogramExtractor

# class WakeWordDataset(Dataset):
#     """
#     A custom PyTorch Dataset that handles the ETL pipeline:
#     Extracts raw audio, Transforms via DSP, and Loads as batched Tensors.
#     """
#     def __init__(self, file_paths, labels, sample_rate=16000, max_duration_sec=1.0):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.sample_rate = sample_rate
#         # Calculate exactly how many raw samples equal 1 second of audio
#         self.max_length = int(sample_rate * max_duration_sec)
        
#         # Initialize our custom hybrid DSP module
#         self.feature_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)

#     def __len__(self):
#         """Returns the total number of samples in the dataset."""
#         return len(self.file_paths)
        
#     def _pad_or_truncate(self, signal):
#         """
#         Forces every continuous audio signal into a strict 1-second boundary.
#         CNNs require fixed-dimension tensors to batch correctly.
#         """
#         signal_length = len(signal)
        
#         if signal_length > self.max_length:
#             # Truncate: Keep the first second, discard the rest
#             return signal[:self.max_length]
#         elif signal_length < self.max_length:
#             # Pad: Add zeros (absolute silence) to the end of the array
#             padding = self.max_length - signal_length
#             return np.pad(signal, (0, padding), 'constant')
            
#         return signal

#     def __getitem__(self, idx):
#         """
#         The core ETL loop. Called dynamically for every file during training.
#         """
#         # 1. Extract: Load the raw 1D audio waveform
#         file_path = self.file_paths[idx]
#         signal, _ = librosa.load(file_path, sr=self.sample_rate)
        
#         # 2. Transform Phase A: Standardize the temporal length
#         signal = self._pad_or_truncate(signal)
        
#         # 3. Transform Phase B: Apply mathematical STFT & Mel scaling
#         # Output shape is a 2D matrix: (n_mels, time_frames)
#         mel_spec = self.feature_extractor.process(signal)
        
#         # 4. Load: Convert to PyTorch Tensor
#         mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        
#         # PyTorch Conv2d requires a channel dimension: (Channels, Height, Width)
#         # Audio is single-channel (mono), so we unsqueeze to make it (1, n_mels, time_frames)
#         mel_tensor = mel_tensor.unsqueeze(0)
        
#         label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
#         return mel_tensor, label_tensor


import torch
import torchaudio
from torch.utils.data import Dataset

class WakeWordDataset(Dataset):
    """
    A custom PyTorch Dataset that handles the ETL pipeline natively in PyTorch.
    Strictly enforces the (1, Mels, Time) DSP convention for visualization and CNN ingestion.
    """
    def __init__(self, file_paths, labels, sample_rate=16000, max_duration_sec=1.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        # Calculate exactly how many raw samples equal 1 second of audio
        self.max_length = int(sample_rate * max_duration_sec)
        
        # Native PyTorch DSP transformations
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)
        
    def _pad_or_truncate(self, waveform):
        """
        Forces every continuous audio signal into a strict 1-second boundary.
        Operates directly on the PyTorch tensor.
        """
        # waveform shape is (Channels, Time)
        num_samples = waveform.shape[1]
        
        if num_samples > self.max_length:
            # Truncate: Keep the first second, discard the rest
            return waveform[:, :self.max_length]
        elif num_samples < self.max_length:
            # Pad: Add zeros (absolute silence) to the end of the tensor
            padding = self.max_length - num_samples
            return torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform

    def __getitem__(self, idx):
        """
        The core ETL loop. Called dynamically for every file during training.
        """
        # 1. Extract: Load via torchaudio (bypasses numpy entirely)
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Guardrail: Resample if the file doesn't match the target sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Guardrail: Force mono if the file is stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 2. Transform Phase A: Standardize the temporal length
        waveform = self._pad_or_truncate(waveform)
        
        # 3. Transform Phase B: Apply mathematical STFT & Mel scaling
        # Output strictly conforms to (1, n_mels, time_frames) -> (1, 40, 98)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amp_to_db(mel_spec)
        
        # 4. Load: Format label and return
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return mel_spec, label_tensor