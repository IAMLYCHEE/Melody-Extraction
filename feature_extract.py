import librosa
import numpy as np
import peak_break

def freq_feature(audio, sr, freq_min=56, freq_max=1760,
                 win_length=1024, hop_length=128,
                 window_shape='hann', n_fft=8192):
    s_stft = librosa.stft(audio, n_fft=n_fft, win_length=win_length,
                          hop_length=hop_length, window=window_shape)
    s_normal = 2 * np.absolute(s_stft) / np.sum(np.hanning(win_length))
    k_min = np.floor(freq_min * n_fft / sr)
    k_max = np.floor(freq_max * n_fft / sr)
    s_crop = s_normal[int(k_min): int(k_max), :]
    return s_crop

def mean_best2peak(s_crop):
    
