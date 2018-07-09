from __future__ import print_function
import numpy as np
import librosa
# this part is to test whether the wave file is loaded correctly
import sys

print('Collecting modules successfully!')

audio_file = './wav/'
if len(sys.argv) != 2:
    print('please input a filename,exiting....')
    sys.exit(2)
else:
    audio_file = audio_file + str(sys.argv[1])
    if audio_file[-4:] != '.wav':
        print('sorry can only deal with wav file now')
        sys.exit(2)


# audio_file = 'meiruoliming.wav'
print('Reading  audio file')
audio, sr = librosa.load(audio_file, sr=48000, mono=True)
# plt.figure()
# plt.plot(audio)


win_length = 1024
window = 'hann'
n_fft = 8192
hop_length = 128
X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, \
                 win_length=win_length, window=window)
X_Normal = 2 * np.absolute(X) / np.sum(np.hanning(win_length))
# plt.figure()
# librosa.display.specshow(librosa.amplitude_to_db(X_Normal,ref=np.max),\

freq_min = 55
freq_max = 1760
k_min = np.floor(freq_min * n_fft / sr)
k_max = np.floor(freq_max * n_fft / sr)
# print(X_Normal.shape)
# print(X_Normal.shape)
X_Crop = X_Normal[int(k_min): int(k_max), :]
# print(X_Crop.shape)

# test librosa salient
freqs = np.linspace(freq_min, freq_max, num=X_Crop.shape[0])
# # freqs test ok
# # print(freqs)
# harms = [1,2,3,4,5]
# weights = [1.0,0.45,0.33,0.25,0.10]
#
# from __future__ import division, print_function
# import numpy as np

print('gathering music information')


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
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


sample_amount = X_Crop.shape[1]
# define the space to store the peak frequencies
pks_locs = np.zeros((sample_amount, 3))

# import the signal package
from scipy import signal

for i in range(sample_amount):
    iFrame = X_Crop[:, i]
    peakInd = detect_peaks(iFrame)
    # plt.plot(X_Crop[:,850])
    # get the amplitude of the peaks
    if (peakInd.size >= 3):
        local_max_Peaks = iFrame[peakInd]
        # sort the amplitudes and find the stornges ones using sort
        indSort = np.argsort(local_max_Peaks)
        # find the frequency according to the index
        sortedLocs = peakInd[indSort]
        sortedLocs = sortedLocs[::-1]
        # put them into the memery space
        # just the first three strongest
        pks_locs[i, :] = sortedLocs[0:3]
    else:
        pks_locs[i, :] = np.array([0, 0, 0])

# this block is to seperate the seqs
hop = 128
frameSize = 1024

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

# print(E.shape)


# this block is to compute zero crossing
zeros_cros = np.zeros((sample_amount, 1))
for i in range(sample_amount):
    seq = seqs[i, :]
    indexs = np.diff(np.sign(seq))
    indexs = np.abs(indexs)
    zeros_pos = np.array((np.where(indexs == 2)))
    zeros_cros[i] = zeros_pos.shape[1]

# S_mfcc = librosa.feature.mfcc(y = audio,sr = sr, hop_length = hop_length, \
#                                         n_fft= n_fft,n_mfcc = 20)
auto_corr = np.zeros((sample_amount, 1))
for i in range(sample_amount):
    seq = seqs[i, :]
    numerator = np.sum(np.multiply(seq[1:frameSize], seq[0:frameSize - 1]))
    denominator = np.sqrt(np.sum(np.multiply(seq[1:frameSize], seq[1:frameSize])) * \
                          np.sum(np.multiply(seq[0:frameSize - 1], seq[0:frameSize - 1])))
    if (denominator == 0):
        denominator = 0.0001;

    auto_corr[i] = numerator / denominator

# transform all to np.array again
E = np.reshape(E, (sample_amount, 1))
zeros_cros = np.reshape(zeros_cros, (sample_amount, 1))
# print(zeros_cros.shape)
auto_corr = np.reshape(auto_corr, (sample_amount, 1))
# print(auto_corr.shape)


pks_locs = np.array(pks_locs)
pks_locs.sort(axis=1)

mean_pks = np.mean(pks_locs[:, 0:2], axis=1)
mean_pks = np.reshape(mean_pks, (sample_amount, 1))
dataSet = np.concatenate((E, zeros_cros, auto_corr, mean_pks), axis=1)

# print(dataSet[1200:1250,:])
# zscore
from scipy import stats

dataSet = stats.zscore(dataSet, axis=0)

from sklearn.cluster import KMeans

start_matrix = np.array([[1, -1, 0, 0],
                         [0, 0, 1, -1],
                         [-2, 2, 2, 2],
                         [-3, 3, 3, 3]])

kmeans = KMeans(n_clusters=4, init=start_matrix, n_init=1).fit(dataSet)
pyidx = kmeans.labels_

# x = np.linspace(0, len(audio), len(audio))
# x = x / 128;
# # plt.figure()
pyidx[np.where(pyidx > 0)] = 1
# plt.plot(x,audio)
diff_pyidx = np.diff(pyidx)
# plt.plot(diff_pyidx)


pitch_on = np.where(diff_pyidx < 0)
pitch_off = np.where(diff_pyidx > 0)
# print(pitch_on[0])
# print(np.where(pitch_off > pitch_on[0,0]))
pitch_off_arr = pitch_off[0]
pitch_on_arr = pitch_on[0]
start = pitch_on_arr[0]
pitch_off_arr = pitch_off_arr[np.where(pitch_off_arr > start)]

# print(pitch_on_arr)
# print(pitch_off_arr)
# find the duration of the small durations, here the threshold is set to 10
# that is approximately 30ms
duration = pitch_off_arr - pitch_on_arr
small_dur_idx = np.where(duration < 15)
pitch_on_arr = np.delete(pitch_on_arr, small_dur_idx)
pitch_off_arr = np.delete(pitch_off_arr, small_dur_idx)


# print(pitch_on_arr)
# print(pitch_off_arr)
# print(len(pitch_on_arr))
# # plt.figure()
def findBreak(energy_wave, window_size=80, hop_size=10, height=2.5):
    wave_dealt = energy_wave.flatten()
    wave_dealt = wave_dealt[window_size // 2: -window_size // 2]
    wave_dealt = stats.zscore(wave_dealt)
    sample_amount = np.floor((len(wave_dealt) - window_size) / hop_size)
    sample_amount = np.int16(sample_amount)
    sum_diff = np.zeros(sample_amount)
    for i in range(sample_amount):
        pos = i * hop_size
        wave_part = wave_dealt[pos: pos + window_size]
        diff_wave = np.diff(wave_part)
        sum_diff[i] = np.sum(diff_wave[0:window_size // 2]) * (-1) \
                      + np.sum(diff_wave[window_size // 2:])
    #     plt.figure()
    #     plt.plot(sum_diff)
    #     plt.figure()
    #     plt.plot(wave_dealt)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sum_diff, height=height)
    final_peaks = np.array([])
    for peak in peaks:
        if (peak < sum_diff.size - 1):
            if sum_diff[peak] > 3.5:
                final_peaks = np.append(final_peaks, peak)
            else:
                diff_area = sum_diff[peak - 1] + sum_diff[peak + 1] + sum_diff[peak]
                if diff_area > 6.0:
                    final_peaks = np.append(final_peaks, peak)
    #     peaks,_ = find_peaks(sum_diff,height = height)
    i = 0
    if final_peaks.size > 1:
        while i < final_peaks.size - 1:
            if (final_peaks[i + 1] - final_peaks[i]) <= 3:
                final_peaks = np.delete(final_peaks, i)
                i -= 1
            i += 1
    return final_peaks


duration = pitch_off_arr - pitch_on_arr
big_dur_idx = np.where(duration > 215)
pitch_on_arr_big_dur = pitch_on_arr[big_dur_idx];
pitch_off_arr_big_dur = pitch_off_arr[big_dur_idx];
# print(pitch_on_arr)
# print(pitch_off_arr)
# print(pitch_on_arr_big_dur)
# print(pitch_off_arr_big_dur)
pitch_off_arr_inter = pitch_off_arr
pitch_on_arr_inter = pitch_on_arr
window_size = 80
hop_size = 10
for i in range(pitch_on_arr_big_dur.size):
    start = pitch_on_arr_big_dur[i]
    end = pitch_off_arr_big_dur[i]
    energy_wave = E[start:end]
    break_idx = findBreak(energy_wave)
    #     print(break_idx.size)
    if break_idx.size > 0:
        for i in range(break_idx.size):
            pitch_off_arr_inter = np.append(pitch_off_arr_inter, start + break_idx[i] * hop_size + window_size)
            pitch_on_arr_inter = np.append(pitch_on_arr_inter,
                                           start + break_idx[i] * hop_size + window_size + 1)

pitch_on_arr = np.sort(pitch_on_arr_inter)
pitch_off_arr = np.sort(pitch_off_arr_inter)

duration = pitch_off_arr - pitch_on_arr
small_dur_idx = np.where(duration < 25)
pitch_on_arr = np.delete(pitch_on_arr, small_dur_idx)
pitch_off_arr = np.delete(pitch_off_arr, small_dur_idx)

pitch_on_arr = np.int16(pitch_on_arr)
pitch_off_arr = np.int16(pitch_off_arr)

pitch_on_arr = np.int16(pitch_on_arr)
pitch_off_arr = np.int16(pitch_off_arr)


print('Audio seperation finished')

freq_hz = 55 + (1760 - 55) / 291 * pks_locs
pitch_midi = 12 * np.log2(32 * freq_hz / 440) + 9

# delete pitch outlier
print('remove pitch outliers .....')
alpha = 2.5
for i in range(len(pitch_on_arr) - 1):
    if i > 0:
        # take out the candidates
        pitch_candi = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i], 0]
        # take out the neighbour of the candidates
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

final_freq = np.zeros(pitch_midi.shape)
for i in range(len(pitch_on_arr)):
    final_freq[pitch_on_arr[i]: pitch_off_arr[i], 0] = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i], 0]

final_freq = final_freq[:, 0]
# x = np.linspace(0, len(audio), len(audio))



print('smooth the music')
from scipy.ndimage.filters import median_filter

# med filter again
final_freq1 = np.zeros(final_freq.shape)
for i in range(len(pitch_on_arr)):
    pitch_candi = final_freq[pitch_on_arr[i]: pitch_off_arr[i]]
    pitch_candi = median_filter(input=pitch_candi, size=93, mode='reflect')
    final_freq1[pitch_on_arr[i]: pitch_off_arr[i]] = pitch_candi

final_freq2 = final_freq1


notes = []
velocity = 95
for i in range(len(pitch_on_arr)):
    #take out the pitches
    pitches = final_freq2[pitch_on_arr[i]: pitch_off_arr[i]]
    duration = 1
    start = pitch_on_arr[i]
    # now assume this is just a solo block
    solo_flag = True
    for j in range(len(pitches) - 1):
        if pitches[j + 1] == pitches[j]:
            # if the pitches are the same, we just continue duration
            duration += 1
        else:
            # if different clearly there exist different block
            solo_flag = False
            # if the accumulated duraion is larger than 40 :0.107s
            if duration > 40:
                #we add the note
                end = duration + start
                #                 pitch = pitches[j]
                pitch_candidates = final_freq2[start:end]
                # the pitch we choose is the mode
                pitch_mode = stats.mode(pitch_candidates)
                pitch = pitch_mode[0][0]
                notes.append((start, end, pitch))
                start = end
                duration = 1
        # Append the rest notes
        if j == (len(pitches) - 1) and duration > 1:
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

for i in range(len(notes)):
    note = notes[i]
    pitch = note[2]
    if i > 0 and i < (len(notes)-1):
        note_pre = notes[i-1]
        note_post = notes[i+1]
        pitch_pre = note_pre[2]
        pitch_post = note_post[2]
        if pitch> (pitch_pre + pitch_post)/2 + 8.5:
            pitch = np.int8(np.round((pitch_pre + pitch_post)/2))
            note_temp = list(note)
            note_temp[2] = pitch
            notes[i] = tuple(note_temp)


final_freq3 = np.zeros(final_freq2.shape)
for note in notes:
    final_freq3[note[0]:note[1]] = note[2]
print('final freq1,freq2,freq3 generated')

# now we use the window with size 35 to get the pitch for every block
notes = []
w_length = 35
for i in range(len(pitch_on_arr)):
    block_length = pitch_off_arr[i] - pitch_on_arr[i]
    w_amount = np.int16(block_length) // np.int16(w_length)
    if w_amount > 0:
        for j in range(w_amount - 1):
            start = pitch_on_arr[i] + j * w_length
            end = start + w_length
            ref_frame = final_freq3[start:end]
            pitch_ref_mode = stats.mode(ref_frame)
            pitch_ref = pitch_ref_mode[0][0]

            curr_frame = final_freq1[start:end]
            pitch_curr_mode = stats.mode(curr_frame)
            pitch_curr = pitch_curr_mode[0][0]
            # compare the reference pitch with the current pitch
            # I set the threshold to be 8.5
            if (pitch_curr > pitch_ref + 8.5) and (pitch_ref != 0):
                pitch = pitch_ref
            else:
                pitch = pitch_curr
            # now append it to the notes we store
            notes.append((start, end, pitch))
        # now deal with the rest of the notes
        start = pitch_on_arr[i] + w_length * (w_amount - 1)
        end = pitch_off_arr[i]
        ref_frame = final_freq3[start:end]
        pitch_ref_mode = stats.mode(ref_frame)
        pitch_ref = pitch_ref_mode[0][0]
        curr_frame = final_freq1[start:end]
        pitch_curr_mode = stats.mode(curr_frame)
        pitch_curr = pitch_curr_mode[0][0]
        if (pitch_curr > pitch_ref + 8.5) and (pitch_ref != 0):
            pitch = pitch_ref
        else:
            pitch = pitch_curr
        notes.append((start, end, pitch))


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

for i in range(len(notes)):
    note = notes[i]
    pitch = note[2]
    if i > 0 and i < (len(notes) - 1):
        note_pre = notes[i - 1]
        note_post = notes[i + 1]
        pitch_pre = note_pre[2]
        pitch_post = note_post[2]
        if pitch > (pitch_pre + pitch_post) / 2 + 8.5:
            pitch = np.int8(np.round((pitch_pre + pitch_post) / 2))
            note_temp = list(note)
            note_temp[2] = pitch
            notes[i] = tuple(note_temp)

final_freq4 = np.zeros(final_freq2.shape)
for note in notes:
    final_freq4[note[0]:note[1]] = note[2]


print('final freq4 generated OK')


modified = 0
for i in range(len(notes)):
    note = notes[i]
    testArray = final_freq[note[0]:note[1]]
    array_fft = np.abs(np.fft.fft(testArray))[1:]
    if np.mean(array_fft) > 45.0:
        sortedArray = np.sort(testArray)
        mean_firsthalf = np.mean(sortedArray[0:len(sortedArray)//2])
        mean_lasthalf = np.mean(sortedArray[len(sortedArray)//2 : ])
        mean_fivelast = np.mean(sortedArray[-8:-3])
        if mean_lasthalf - mean_firsthalf > 8.0:
            note_temp = list(note)
            note_temp[2] = mean_fivelast
            notes[i] = tuple(note_temp)
            modified = modified + 1

print('modified ' + repr(modified) + ' notes')

final_freq5 = np.zeros(final_freq2.shape)
for note in notes:
    final_freq5[note[0]:note[1]] = note[2]



def pitch2freq(pitch):
    return 440 / 32 * np.power(2, (pitch - 9)/12)


from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play

C = 128.0 / 48.0
note_start = notes[0]
start_time = note_start[0] * C
sound = Sine(0).to_audio_segment(duration=start_time)

# print(sound.duration_seconds)
silenceCount = 0
for i in range(len(notes) - 1):
#     print(i)
    noteCurr = notes[i]
    notePost = notes[i + 1]
    durationBetween = (notePost[0] - noteCurr[1])
    if durationBetween == 0:
        durationCurr = (noteCurr[1] - noteCurr[0]) * C
        pitch = noteCurr[2]
        freq = pitch2freq(pitch)
        tone = Sine(freq,sample_rate = 48000).to_audio_segment(duration=durationCurr + 25)
        sound = sound.append(tone, crossfade=25)
#         print(sound.duration_seconds)
    else:
        silenceCount = silenceCount + 1
        durationCurr = (noteCurr[1] - noteCurr[0]) * C
        pitch = noteCurr[2]
        freq = pitch2freq(pitch)
        tone = Sine(freq,sample_rate = 48000).to_audio_segment(duration=durationCurr + 25)
        sound = sound.append(tone, crossfade=25)
#         print(sound.duration_seconds)
        silenceStart = noteCurr[1]
        silenceEnd = notePost[0]
        silenceDuration = (silenceEnd - silenceStart) * C
        tone = Sine(0,sample_rate = 48000).to_audio_segment(duration=silenceDuration + 15)
#         print(silenceStart)
        sound = sound.append(tone, crossfade=15)

# add the last note
noteLast = notes[-1]
durationLast = (noteLast[1] - noteLast[0])*C
freq = pitch2freq(noteLast[2])
tone = Sine(freq).to_audio_segment(duration=durationLast + 25)
sound = sound.append(tone, crossfade=25)
print(sound.duration_seconds)


sound2 = AudioSegment.from_wav(audio_file)
# play(sound2)
print(len(sound2))
silence = AudioSegment.silent(duration=len(sound2)+100)
left = silence.overlay(sound, gain_during_overlay=-8)
right = silence.overlay(sound2, gain_during_overlay=-8)
stereo_sound = AudioSegment.from_mono_audiosegments(left,right)
filename = './wav2wavmix/'+audio_file[6:-4] + '_mix_conti_v4.wav'
stereo_sound.export(filename,format="wav",bitrate="48k")

print('stereo sound file generated!')