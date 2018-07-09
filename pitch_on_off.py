# pitch_on_off.py
# Li Yicheng, t-yicli@microsoft.com
import numpy as np
import peak_break
from sklearn.cluster import KMeans


def gen_pitch_on_off(dataset,energy, small_dur=15, large_dur = 215,window_size = 80,hop_size = 10):
    # step 1: using the kmeans to seperate first
    start_matrix = np.array([[1, -1, 0, 0],
                             [0, 0, 1, -1],
                             [-2, 2, 2, 2],
                             [-3, 3, 3, 3]])

    kmeans = KMeans(n_clusters=4, init=start_matrix, n_init=1).fit(dataset)
    pyidx = kmeans.labels_
    # set the class labels larger than 0 to be 1
    pyidx[np.where(pyidx > 0)] = 1
    # calculate the derivative of the labels
    diff_pyidx = np.diff(pyidx)
    # pitch on is when derivate to be negative, that is from class 1 to class 0
    pitch_on = np.where(diff_pyidx < 0)
    # pitch off is when derivate to be positive, that is from class 0 to class 1
    pitch_off = np.where(diff_pyidx > 0)
    # get the np array of the on and off index
    pitch_off_arr = pitch_off[0]
    pitch_on_arr = pitch_on[0]
    # align the time line
    start = pitch_on_arr[0]
    pitch_off_arr = pitch_off_arr[np.where(pitch_off_arr > start)]

    # step2: Optimization
    # find the duration of the small durations, here the threshold is set to 10
    # that is approximately 30ms
    duration = pitch_off_arr - pitch_on_arr
    small_dur_idx = np.where(duration < small_dur)
    pitch_on_arr = np.delete(pitch_on_arr, small_dur_idx)
    pitch_off_arr = np.delete(pitch_off_arr, small_dur_idx)

    # add more breaks in the large duration

    duration = pitch_off_arr - pitch_on_arr
    big_dur_idx = np.where(duration > large_dur)
    pitch_on_arr_big_dur = pitch_on_arr[big_dur_idx]
    pitch_off_arr_big_dur = pitch_off_arr[big_dur_idx]
    pitch_off_arr_inter = pitch_off_arr
    pitch_on_arr_inter = pitch_on_arr
    for i in range(pitch_on_arr_big_dur.size):
        start = pitch_on_arr_big_dur[i]
        end = pitch_off_arr_big_dur[i]
        energy_wave = energy[start:end]
        break_idx = peak_break.find_break(energy_wave=energy_wave)
        if break_idx.size > 0:
            for j in range(break_idx.size):
                pitch_off_arr_inter = np.append(pitch_off_arr_inter, start + break_idx[j] * hop_size + window_size)
                pitch_on_arr_inter = np.append(pitch_on_arr_inter,
                                               start + break_idx[j] * hop_size + window_size + 1)

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

    return pitch_on_arr,pitch_off_arr