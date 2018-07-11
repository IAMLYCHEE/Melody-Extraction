# app.py
# Li Yicheng t-yicli@microsoft.com


import feature_extract
import pitch_on_off
import user_input
import librosa
import pitch_extract
import utilities
import peak_break


print('Collecting modules Successfully')
audio_file,mode = user_input.cmd_input()
audio, sr = librosa.load(audio_file, sr = 48000, mono = True)


break_points,hop = peak_break.detect_audio_break(audio,sr=48000,frame_size = 48000, hop_size = 8000 ,threshold = 2)
# print(break_points)
audio_pieces = []
if len(break_points) > 0:
    curr = 0

    for break_point in break_points:
        audio_piece = audio[curr * hop : break_point*hop]
        curr = break_point
        audio_pieces.append(audio_piece)
    audio_piece_last = audio[curr*hop:]    
    audio_pieces.append(audio_piece_last)
else:
    audio_pieces.append(audio)

audio_notes = []
i = 0
hop = 8000
hop_length = 128
for audio_piece in audio_pieces:
    audio = audio_piece
    s_crop = feature_extract.freq_feature(audio, sr, freq_min=56, freq_max=1760,
                                      win_length=1024, hop_length=128,
                                      window_shape='hann', n_fft=8192)
    sample_amount = s_crop.shape[1]
    seqs = feature_extract.sep_audio(audio, sample_amount,hop_length=128,win_length=1024)
    energy = feature_extract.compute_energy(seqs)
    zeros_cros = feature_extract.compute_zerocross(seqs)
    auto_corr = feature_extract.compute_autocorr(seqs)
    mean_pks,pks_locs = feature_extract.mean_best2peak(s_crop)
    dataset = feature_extract.gen_dataSet(energy=energy,zeros_cros=zeros_cros,
                                          auto_corr=auto_corr,mean_pks=mean_pks,
                                          sample_amount=sample_amount)
    pitch_on_array, pitch_off_array = pitch_on_off.gen_pitch_on_off(dataset=dataset,energy=energy,
                                                                    small_dur=15,large_dur=215,
                                                                    window_size=80,hop_size=10)
    notes = pitch_extract.pitch_extract(pks_locs,pitch_on_array,pitch_off_array,
                                        suppress_alpha=2.5,n_fft=8192,sr=48000,freq_min=55,
                                        outlier = 8.5, detect_osc = 40.0, jump = 7.0)
    for note in notes:
        if i>0:
            start = break_points[i-1] * hop //hop_length + note[0]
            end = break_points[i-1]*hop//hop_length + note[1]
            audio_notes.append((start,end,note[2]))
        else:
            audio_notes.append(note)         
    i+=1

if(mode == 'wav2wav'):
    utilities.genStereo(audio_notes,audio_file,hop_length=128,sr = 48000,crossfade=25,
                        silencefade=15)
elif (mode == 'wav2midi'):
    utilities.genMidi(audio_notes,audio_file,initial_tempo=80,program=42)
else:
    utilities.genStereo(audio_notes,audio_file,hop_length=128,sr = 48000,crossfade=25,
                        silencefade=15)
    utilities.genMidi(audio_notes, audio_file, initial_tempo=80, program=42)