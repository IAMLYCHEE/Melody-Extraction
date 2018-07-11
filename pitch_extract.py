# pitch_extract.py
# Li Yicheng t-yicli@microsoftl.com
import utilities
import numpy as np
from scipy import stats


def pitch_extract(pks_locs,pitch_on_arr,pitch_off_arr,
                  suppress_alpha=2.5,n_fft=8192,sr=48000,freq_min=55,
                  outlier = 8.5, detect_osc = 45.0, jump = 8.0):
    freq_hz = utilities.pks2freq(pks_locs[:,0],n_fft=n_fft,sr = sr,
                                 freq_min=freq_min)
    pitch_midi = utilities.freq2midi(freq_hz)
    alpha = suppress_alpha
    for i in range(len(pitch_on_arr) - 1):
        if i > 0:
            # take out the candidates
            pitch_candi = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i]]
            # take out the neighbour of the candidates
            pitch_candi_pre = pitch_midi[pitch_on_arr[i - 1]: pitch_off_arr[i - 1]]
            pitch_candi_post = pitch_midi[pitch_on_arr[i + 1]: pitch_off_arr[i + 1]]
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
            pitch_midi[pitch_on_arr[i]: pitch_off_arr[i]] = pitch_candi

    final_freq = np.zeros(pitch_midi.shape)
    for i in range(len(pitch_on_arr)):
        final_freq[pitch_on_arr[i]: pitch_off_arr[i]] = pitch_midi[pitch_on_arr[i]: pitch_off_arr[i]]

    print('smooth the music')
    from scipy.ndimage.filters import median_filter

    # med filter again
    final_freq1 = np.zeros(final_freq.shape)
    for i in range(len(pitch_on_arr)):
        pitch_candi = final_freq[pitch_on_arr[i]: pitch_off_arr[i]]
        pitch_candi = median_filter(input=pitch_candi, size=93, mode='reflect')
        final_freq1[pitch_on_arr[i]: pitch_off_arr[i]] = pitch_candi

    final_freq2 = final_freq1


    print('pitch contouring...')

    notes = []
    for i in range(len(pitch_on_arr)):
        # take out the pitches
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
                    # we add the note
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
        if i > 0 and i < (len(notes) - 1):
            note_pre = notes[i - 1]
            note_post = notes[i + 1]
            pitch_pre = note_pre[2]
            pitch_post = note_post[2]
            if pitch > (pitch_pre + pitch_post) / 2 + outlier:
                pitch = (pitch_pre + pitch_post) / 2
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
        w_amount = np.int32(block_length) // np.int32(w_length)
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
                if (pitch_curr > pitch_ref + outlier) and (pitch_ref != 0):
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
            if (pitch_curr > pitch_ref + outlier) and (pitch_ref != 0):
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
            if pitch > (pitch_pre + pitch_post) / 2 + outlier:
                pitch = (pitch_pre + pitch_post) / 2
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
        sortedArray = np.sort(testArray)
        mean_firsthalf = np.mean(sortedArray[0:len(sortedArray)//2])
        mean_lasthalf = np.mean(sortedArray[len(sortedArray)//2 : ])
        array_length = len(sortedArray)
        subIndex1 = np.int32(array_length * 0.7)
        subIndex2 = np.int32(array_length * 0.85)
        mean_fivelast = np.mean(sortedArray[subIndex1:subIndex2])
        if mean_lasthalf - mean_firsthalf > jump or np.mean(array_fft) > detect_osc:
            note_temp = list(note)
            note_temp[2] = mean_fivelast
            notes[i] = tuple(note_temp)
            modified = modified + 1

    print('modified ' + repr(modified) + ' notes')

    final_freq5 = np.zeros(final_freq2.shape)
    for note in notes:
        final_freq5[note[0]:note[1]] = note[2]

    return notes