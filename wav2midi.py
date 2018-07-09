# author Yicheng Li, t-yicli@microsoft.com
from __future__ import division, print_function

import numpy as np
# import scipy.io as sio


import sys

audio_file = 'vocal_short.wav'
# if len(sys.argv) != 2:
#     print('please input a filename')
#     sys.exit(2)
# else:
#     audio_file = str(sys.argv[1])

plot_on = False

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind


# wrap up
import librosa
# import librosa.display

# load wav file
# filename = 'choubaguai'
# audio_file = filename + '.wav'
audio, sr = librosa.load(audio_file, sr=48000, mono=True)

# STFT to get the pitch
win_length = 1024
window = 'hann'
n_fft = 8192
hop_length = 128
X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                 win_length=win_length, window=window)
X_Normal = 2 * np.absolute(X) / np.sum(np.hanning(win_length))

# Crop the data to increase the speed
freq_min = 55
freq_max = 1760
k_min = np.floor(freq_min * n_fft / sr)
k_max = np.floor(freq_max * n_fft / sr)
X_Crop = X_Normal[int(k_min): int(k_max), :]

# now we perform the salience function
# freqs = np.linspace(freq_min, freq_max, num=X_Crop.shape[0])
# harms = [1, 2, 3, 4, 5]
# weights = [1.0, 0.45, 0.33, 0.25, 0.10]
# S_sal = librosa.salience(X_Crop, freqs, harms, weights, fill_value=0)

# now is to get the location of the peaks
sample_amount = X_Crop.shape[1]
# define the space to store the peak frequencies
pks_locs = np.zeros((sample_amount, 3))

# import the signal package

for i in range(sample_amount):
    iFrame = X_Crop[:, i]
    peakInd = detect_peaks(iFrame)
    local_max_Peaks = iFrame[peakInd]
    # sort the amplitudes and find the stornges ones using sort
    indSort = np.argsort(local_max_Peaks)
    # find the frequency according to the index
    sortedLocs = peakInd[indSort]
    sortedLocs = sortedLocs[::-1]
    # put them into the memery space
    # just the first three strongest
    if sortedLocs.shape[0] >= 3:
        pks_locs[i, :] = sortedLocs[0:3]
    else:
        pks_locs[i, :] = np.array([0, 0, 0])

# this block is to seperate the seqs
hop = 128
frameSize = 1024

print('gathering information')
# now create a space to store the seqs
seqs = np.zeros((sample_amount, frameSize))
# Now we put the blocks into the seqs
for i in np.arange(0, len(audio) - frameSize, hop):
    seqs[i // 128, :] = audio[i:i + frameSize]

# this block is to compute energy
E = np.multiply(seqs, seqs)
E = np.sum(E, axis=1)
E[np.where(E < 0.0000001)] = 0.00001
E = np.log10(E)

# this block is to compute zero crossing amount
# assign the memory space (sample_amount,1)
zeros_cros = np.zeros((sample_amount, 1))
for i in range(sample_amount):
    seq = seqs[i, :]
    indexs = np.diff(np.sign(seq))
    indexs = np.abs(indexs)
    zeros_pos = np.array((np.where(indexs == 2)))
    zeros_cros[i] = zeros_pos.shape[1]

# this block is to compute the auto correlation
# assign the memory space (sample_amount,1)
auto_corr = np.zeros((sample_amount, 1))
for i in range(sample_amount):
    seq = seqs[i, :]
    numerator = np.sum(np.multiply(seq[1:frameSize], seq[0:frameSize - 1]))
    denominator = np.sqrt(np.sum(np.multiply(seq[1:frameSize], seq[1:frameSize])) *
                          np.sum(np.multiply(seq[0:frameSize - 1], seq[0:frameSize - 1])))
    if denominator == 0:
        denominator = 0.0001
    auto_corr[i] = numerator / denominator

# this block is to concatenate all the features to form the dataset
# transform all to np.array again
E = np.reshape(E, (sample_amount, 1))
zeros_cros = np.reshape(zeros_cros, (sample_amount, 1))
auto_corr = np.reshape(auto_corr, (sample_amount, 1))
# I am only interested in the first 3 peak locs
pks_locs = pks_locs[:, 0:2]
# after some experiments I found out it is better to sort the pks_locs somehow
pks_locs.sort(axis=1)

# concatenate along the features
dataSet = np.concatenate((E, zeros_cros, auto_corr, pks_locs[:, 0:2]), axis=1)

# now perform the zscore, because we would use kmeans to cluster
from scipy import stats

dataSet = stats.zscore(dataSet, axis=0)

# now is to clustering
from sklearn.cluster import KMeans

print('find initial breaks')
start_matrix = np.array([[1, -1, 3, 0, 0],
                         [0, 0, 2, -1, -1],
                         [-2, 2, 1, 2, 1],
                         [-3, 3, 0, 3, 3]])
# clustering
kmeans = KMeans(n_clusters=4, init=start_matrix, n_init=1).fit(dataSet)
# generate the labels
pyidx = kmeans.labels_
# let the label above 1 to be 1, the expected sound place is labeled zero
pyidx[np.where(pyidx > 0)] = 1
# differentiate the label
diff_pyidx = np.diff(pyidx)


# define the function to find the break
def findBreak(energy_wave, window_size=80, hop_size=10, height=2.5):
    wave_dealt = energy_wave.flatten()
    # I cut the beginning and part of the end to care about only the middle part
    wave_dealt = wave_dealt[window_size // 2: -window_size // 2]
    # use zscore to normalize
    wave_dealt = stats.zscore(wave_dealt)
    # define the amount of intervals
    sample_amount = np.floor((len(wave_dealt) - window_size) / hop_size)
    sample_amount = np.int16(sample_amount)
    # define a memory space to store the result
    sum_diff = np.zeros(sample_amount)
    for i in range(sample_amount):
        pos = i * hop_size
        wave_part = wave_dealt[pos: pos + window_size]
        diff_wave = np.diff(wave_part)
        # calculate the magnitude of existing a concave
        sum_diff[i] = np.sum(diff_wave[0:window_size // 2]) * (-1) \
                      + np.sum(diff_wave[window_size // 2:])

    from scipy.signal import find_peaks
    # find peaks which height is large than height, in the concave, peak is the index
    peaks, _ = find_peaks(sum_diff, height=height)
    final_peaks = np.array([])
    for peak in peaks:
        if (peak < sum_diff.size - 1):
            if sum_diff[peak] > 3.5:  # magnitude satisfied put it in
                final_peaks = np.append(final_peaks, peak)
            else:  # area satisfied, put it in
                diff_area = sum_diff[peak - 1] + sum_diff[peak + 1] + sum_diff[peak]
                if diff_area > 6.0:
                    final_peaks = np.append(final_peaks, peak)

    # delete the continuous peaks which is really rare
    i = 0
    if final_peaks.size > 1:
        while i < final_peaks.size - 1:
            if (final_peaks[i + 1] - final_peaks[i]) <= 3:
                final_peaks = np.delete(final_peaks, i)
                i -= 1
            i += 1
    return final_peaks


# if the diff is -1 it means it is from noise to voice, now pitch on
pitch_on = np.where(diff_pyidx < 0)
# if the diff is 1, it means it is from voice to noise , now pitch off
pitch_off = np.where(diff_pyidx > 0)
# get the pitch off array
pitch_off_arr = pitch_off[0]
# get the pitch on array
pitch_on_arr = pitch_on[0]
start = pitch_on_arr[0]
# find the first pitch off afgter pitch on
pitch_off_arr = pitch_off_arr[np.where(pitch_off_arr > start)]

print('add mmore breaks')
# this block is to remove the small duration
duration = pitch_off_arr - pitch_on_arr
small_dur_idx = np.where(duration < 15)
pitch_on_arr = np.delete(pitch_on_arr, small_dur_idx)
pitch_off_arr = np.delete(pitch_off_arr, small_dur_idx)

# this block is to find the whether there is break in large duration
duration = pitch_off_arr - pitch_on_arr
big_dur_idx = np.where(duration > 215)
pitch_on_arr_big_dur = pitch_on_arr[big_dur_idx]
pitch_off_arr_big_dur = pitch_off_arr[big_dur_idx]
pitch_off_arr_inter = pitch_off_arr
pitch_on_arr_inter = pitch_on_arr
# define the window size to find the break
window_size_detect = 80
# define the hop size to find the break
hop_size_detect = 10

# add the finded new block into the previous pitch on/off array
for i in range(pitch_on_arr_big_dur.size):
    start = pitch_on_arr_big_dur[i]
    end = pitch_off_arr_big_dur[i]
    energy_wave = E[start:end]
    break_idx = findBreak(energy_wave)
    #     print(break_idx.size)
    if break_idx.size > 0:
        for j in range(break_idx.size):
            pitch_off_arr_inter = np.append(pitch_off_arr_inter, start \
                                            + break_idx[j] * hop_size_detect + window_size_detect)
            pitch_on_arr_inter = np.append(pitch_on_arr_inter, start \
                                           + break_idx[j] * hop_size_detect + window_size_detect \
                                           + window_size_detect // 2)

pitch_on_arr = np.sort(pitch_on_arr_inter)
pitch_off_arr = np.sort(pitch_off_arr_inter)

# remove the small interval again
# this block is to remove the small duration
duration = pitch_off_arr - pitch_on_arr
small_dur_idx = np.where(duration < 25)
pitch_on_arr = np.delete(pitch_on_arr, small_dur_idx)
pitch_off_arr = np.delete(pitch_off_arr, small_dur_idx)

pitch_on_arr = np.int16(pitch_on_arr)
pitch_off_arr = np.int16(pitch_off_arr)

freq_hz = 55 + (1760 - 55) / 291 * pks_locs
pitch_midi = 12 * np.log2(32 * freq_hz / 440) + 9

# delete pitch outlier
print('remove pitch outliers .....')
alpha = 2.5
for i in range(len(pitch_on_arr) - 1):
    if i > 0:
        # take out the candidates
        pitch_candi = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i], 0]
        pitch_candi_pre = pitch_midi[pitch_on_arr[i - 1]: pitch_off_arr[i - 1], 0]
        pitch_candi_post = pitch_midi[pitch_on_arr[i + 1]: pitch_off_arr[i + 1], 0]
        pitches = np.concatenate((pitch_candi_pre, pitch_candi, pitch_candi_post))

        # calculate teh mean of the candidates
        pitch_mean = pitches.mean()
        # calculate the standard variance
        pitch_std = pitches.std()
        # set pitches that is far away from the mean to be zero
        for j in range(len(pitch_candi)):
            if (pitch_candi[j] - pitch_mean) > (alpha * pitch_std):
                pitch_candi[j] = 0

        # calculate the mean of the rest pitches
        pitch_sum = 0
        amount = 0
        for pitch in pitch_candi:
            if pitch != 0:
                pitch_sum += pitch
                amount += 1
        # end
        if amount == 0:
            amount = 1
            print(repr(i) + 'th has some problem at' + repr(pitch_on_arr[i]))
        pitch_mean = pitch_sum / amount

        for j in range(len(pitch_candi)):
            if pitch_candi[j] == 0:
                pitch_candi[j] = pitch_mean

        # modify them to original pitch
        pitch_midi[pitch_on_arr[i]: pitch_off_arr[i], 0] = pitch_candi

# this is designed for the clearance of plot if we use 0 than the vertical lines would sabotage the plot
final_freq = np.zeros(pitch_midi.shape)
for i in range(len(pitch_on_arr)):
    final_freq[pitch_on_arr[i]: pitch_off_arr[i], 0] = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i], 0]

final_freq = final_freq[:, 0]

# %matplotlib qt


if plot_on:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(final_freq)

print('smooth the music')
from scipy.ndimage.filters import median_filter
#med filter again
final_freq1 = np.zeros(final_freq.shape)
for i in range(len(pitch_on_arr)):
    pitch_candi = final_freq[pitch_on_arr[i] : pitch_off_arr[i]]
    pitch_candi = median_filter(input = pitch_candi,size = 93, mode = 'reflect')
    final_freq1[pitch_on_arr[i]: pitch_off_arr[i]] = pitch_candi

if plot_on:
    plt.plot(final_freq1)

print('round the pitches')
final_freq2 = np.round(final_freq1)

if plot_on:
    plt.plot(final_freq2)

notes = []
velocity = 95
for i in range(len(pitch_on_arr)):
    pitches = final_freq2[pitch_on_arr[i]: pitch_off_arr[i]]
    duration = 1
    start = pitch_on_arr[i]
    solo_flag = True
    for j in range(len(pitches) - 1):
        if pitches[j + 1] == pitches[j]:
            duration += 1
        else:
            solo_flag = False
            if duration > 40:
                end = duration + start
                #                 pitch = pitches[j]
                pitch_candidates = final_freq2[start:end]
                pitch_mode = stats.mode(pitch_candidates)
                pitch = pitch_mode[0][0]
                notes.append((start, end, pitch))
                start = end
                duration = 1
        # Append the rest notes
        if j == (len(pitches) - 2) and duration > 1:
            #             end = duration + start
            if duration > 40:
                pitch = pitches[j]
                notes.append((start, start + duration, pitch))
            else:
                if solo_flag:
                    pitch = pitches[j]
                    notes.append((start, start + duration, pitch))
                else:
                    noteTemp = list(notes[-1])
                    noteTemp[-2] += duration
                    notes[-1] = tuple(noteTemp)

i = 0
while i < (len(notes) - 1):
    note_pre = notes[i]
    note_post = notes[i + 1]
    if note_pre[2] == note_post[2] and note_pre[1] == note_post[0]:
        note_pre_temp = list(note_pre)
        note_pre_temp[1] = note_post[1]
        note_pre = tuple(note_pre_temp)
        notes[i] = note_pre
        del notes[i + 1]
        i -= 1
    i += 1

import pretty_midi as pmidi

pm = pmidi.PrettyMIDI(initial_tempo=80)
inst = pmidi.Instrument(program=42, is_drum=False, name='cello')
pm.instruments.append(inst)

velocity = 95
for i in range(len(notes)):
    note = notes[i]
    pitch = np.int8(note[2])
    if i > 0 and i < (len(notes)-1):
        note_pre = notes[i-1]
        note_post = notes[i+1]
        pitch_pre = note_pre[2]
        pitch_post = note_post[2]
        if pitch> (pitch_pre + pitch_post)/2 + 8.5:
            pitch = np.int8(np.round((pitch_pre + pitch_post)/2))
    start_time = note[0] * 128 / 48000
    end_time = note[1] * 128 / 48000
    print(start_time,end_time)
    inst.notes.append(pmidi.Note(velocity, pitch, start_time, end_time))

print('generate ' + repr(len(notes)) + ' notes')
print('write into midi file.....')
print(pm.instruments)
filename = audio_file[0:-4]
pm.write(filename + '_4_0.mid')

# test part
final_freq3 = np.zeros(final_freq2.shape)
for note in notes:
    final_freq3[note[0]:note[1]] = note[2]
if plot_on:
    plt.plot(final_freq3)
    plt.show()

