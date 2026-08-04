

import librosa
from torch.utils.data import Dataset
import torch 
import os 
import numpy as np


class ESC50Dataset(Dataset):
    def __init__(self, dataframe, audio_dir, transform=None,augment=True):
        self.dataframe = dataframe 
        self.audio_dir = audio_dir
        self.transform = transform
        self.audio_files = self.dataframe["filename"].tolist()
        self.augment = augment
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        wave, sr = librosa.load(os.path.join(self.audio_dir,self.audio_files[idx]),sr=16000)
        if self.augment:
            # pass
            noise = np.random.normal(0, 0.005, wave.shape)
            shift = np.random.randint(-0.1*len(wave), 0.1*len(wave))
            wave = np.roll(wave,shift)
            wave = wave + noise  
        mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=128,hop_length=625)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        x_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
        y = self.dataframe.iloc[idx]["target"]
        x_tensor = x_tensor.unsqueeze(0) 
        if self.transform:
            x_tensor = self.transform(x_tensor)
        return x_tensor, torch.tensor(y, dtype=torch.long)
