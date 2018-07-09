#feature_extraction.py
# Li Yicheng t-yicli@microsoft.com
import librosa
import numpy as np
import peak_break
from scipy import stats

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
    sample_amount = s_crop.shape[1]
    pks_locs = np.zeros((sample_amount,3))
    for i in range(sample_amount):
        iFrame = s_crop[:, i]
        peakInd = peak_break.detect_peaks(iFrame)
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

    pks_locs = np.array(pks_locs)
    pks_locs.sort(axis=1)

    mean_pks = np.mean(pks_locs[:, 0:2], axis=1)
    return mean_pks,pks_locs


def sep_audio(audio, sample_amount, hop_length = 128, win_length = 1024):
    seqs = np.zeros((sample_amount, win_length))
    # Now we put the blocks into the seqs
    for i in np.arange(0, len(audio) - win_length, hop_length):
        seqs[i // hop_length, :] = audio[i:i + win_length]
    return seqs


def compute_energy(seqs):
    energy = np.multiply(seqs, seqs)
    energy = np.sum(energy, axis=1)
    energy[np.where(energy < 0.0000001)] = 0.00001
    energy = np.log10(energy)
    return energy


def compute_zerocross(seqs):
    zeros_cros = np.zeros((seqs.shape[0], 1))
    for i in range(seqs.shape[0]):
        seq = seqs[i, :]
        indexs = np.diff(np.sign(seq))
        indexs = np.abs(indexs)
        zeros_pos = np.array((np.where(indexs == 2)))
        zeros_cros[i] = zeros_pos.shape[1]
    return zeros_cros


def compute_autocorr(seqs):
    auto_corr = np.zeros((seqs.shape[0], 1))
    frameSize = seqs.shape[1]
    for i in range(seqs.shape[0]):
        seq = seqs[i, :]
        numerator = np.sum(np.multiply(seq[1:frameSize], seq[0:frameSize - 1]))
        denominator = np.sqrt(np.sum(np.multiply(seq[1:frameSize], seq[1:frameSize])) * \
                              np.sum(np.multiply(seq[0:frameSize - 1], seq[0:frameSize - 1])))
        if (denominator == 0):
            denominator = 0.0001

        auto_corr[i] = numerator / denominator
    return auto_corr


def gen_dataSet(energy, zeros_cros, auto_corr,mean_pks, sample_amount):
    mean_pks = np.reshape(mean_pks, (sample_amount, 1))
    energy = np.reshape(energy, (sample_amount, 1))
    zeros_cros = np.reshape(zeros_cros, (sample_amount, 1))
    auto_corr = np.reshape(auto_corr, (sample_amount, 1))
    dataset = np.concatenate((energy, zeros_cros, auto_corr, mean_pks), axis=1)
    dataset = stats.zscore(dataset, axis=0)
    return dataset
