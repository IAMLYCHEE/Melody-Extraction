# peak_break.py
# Li Yicheng, t-yicli@microsoft.com
import numpy as np
from scipy import stats

def detect_audio_break(audio,sr=48000,frame_size = 48000, hop_size = 8000 ,threshold = 2):
    # the variable we need to return
    break_points = []
    frameSize = frame_size 
    hop = hop_size
    #if the audio is large than 1 minute we cut it 
    if len(audio) > 60 * sr : 
        #calculate the amount of frames 
        sample_amount = (len(audio) - frameSize )//hop + 1
        # now create a space to store the seqs
        seqs = np.zeros((sample_amount, frameSize))
        # Now we put the blocks into the seqs
        for i in np.arange(0, len(audio) - frameSize, hop):
            seqs[i // hop, :] = audio[i:i + frameSize]
        #calculate the energy
        E = np.multiply(seqs, seqs)
        E = np.sum(E, axis=1)    
        # find the small energy
        indexs = np.where(E< threshold)[0]
        # add 0 at the end, for the convenience of the following operation
        indexs = np.append(indexs,[0])
        i = 0
        # a set of store set of positions
        pos_sets = []
        # a set of positions
        pos_set = []
        while i < len(indexs)-1:
            # if continuous add to a set
            if(indexs[i] == indexs[i+1] - 1):
                pos_set.append(indexs[i])
            else:
            # if not continuous, add current to set and create a new set
                pos_set.append(indexs[i])
                pos_sets.append(pos_set)
                pos_set = []
            i = i+1

        for pos_set in pos_sets:
            if len(pos_set) >= 6:
                break_candi = pos_set[len(pos_set)//2]
                if break_candi > sample_amount* 0.3 and break_candi < sample_amount * 0.70:
                    break_points.append(break_candi)
                
    return break_points,hop
    


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


def find_break(energy_wave, window_size=80, hop_size=10, height=2.5):
    wave_dealt = energy_wave.flatten()
    wave_dealt = wave_dealt[window_size // 2: -window_size // 2]
    wave_dealt = stats.zscore(wave_dealt)
    sample_amount = np.floor((len(wave_dealt) - window_size) / hop_size)
    sample_amount = np.int32(sample_amount)
    sum_diff = np.zeros(sample_amount)
    for i in range(sample_amount):
        pos = i * hop_size
        wave_part = wave_dealt[pos: pos + window_size]
        diff_wave = np.diff(wave_part)
        sum_diff[i] = np.sum(diff_wave[0:window_size // 2]) * (-1)\
            + np.sum(diff_wave[window_size // 2:])
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
    i = 0
    if final_peaks.size > 1:
        while i < final_peaks.size - 1:
            if (final_peaks[i + 1] - final_peaks[i]) <= 3:
                final_peaks = np.delete(final_peaks, i)
                i -= 1
            i += 1
    return final_peaks
