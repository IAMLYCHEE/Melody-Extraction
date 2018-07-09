import feature_extract
import peak_break
import user_input
import librosa

print('Collecting modules Successfully')
audio_file = user_input.cmd_input()
audio, sr = librosa.load(audio_file, sr = 48000, mono = True)
s_crop = feature_extract.freq_feature(audio, sr, freq_min=56, freq_max=1760,
                                      win_length=1024, hop_length=128,
                                      window_shape='hann', n_fft=8192)
sample_amount = s_crop.shape[1]







