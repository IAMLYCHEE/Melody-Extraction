# utilities.py
# Li Yicheng t-yicli@microsoft.com
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import pretty_midi as pmidi


def pks2freq(pks_locs,freq_min=55,sr=48000,n_fft=8192):
    return freq_min + sr/n_fft * pks_locs


def freq2midi(freq_hz):
    return 12 * np.log2(32 * freq_hz / 440) + 9


def pitch2freq(pitch):
    return 440 / 32 * np.power(2, (pitch - 9) / 12)

def genStereo(notes,audio_file,hop_length=128,sr=48000,crossfade = 25,silencefade = 15):
    C = hop_length * 1000.0 /sr
    note_start = notes[0]
    start_time = note_start[0] * C
    sound = Sine(0).to_audio_segment(duration=start_time)

    silenceCount = 0
    for i in range(len(notes) - 1):
        noteCurr = notes[i]
        notePost = notes[i + 1]
        durationBetween = (notePost[0] - noteCurr[1])
        if durationBetween == 0:
            durationCurr = (noteCurr[1] - noteCurr[0]) * C
            pitch = noteCurr[2]
            freq = pitch2freq(pitch)
            tone = Sine(freq, sample_rate=sr).to_audio_segment(duration=durationCurr + crossfade)
            sound = sound.append(tone, crossfade=crossfade)
        else:
            silenceCount = silenceCount + 1
            durationCurr = (noteCurr[1] - noteCurr[0]) * C
            pitch = noteCurr[2]
            freq = pitch2freq(pitch)
            tone = Sine(freq, sample_rate=sr).to_audio_segment(duration=durationCurr + crossfade)
            sound = sound.append(tone, crossfade=crossfade)
            silenceStart = noteCurr[1]
            silenceEnd = notePost[0]
            silenceDuration = (silenceEnd - silenceStart) * C
            tone = Sine(0, sample_rate=sr).to_audio_segment(duration=silenceDuration + silencefade)
            sound = sound.append(tone, crossfade=silencefade)

    # add the last note
    noteLast = notes[-1]
    durationLast = (noteLast[1] - noteLast[0]) * C
    freq = pitch2freq(noteLast[2])
    tone = Sine(freq).to_audio_segment(duration=durationLast + crossfade)
    sound = sound.append(tone, crossfade=crossfade)
    # print(sound.duration_seconds)

    sound2 = AudioSegment.from_wav(audio_file)
    # print(len(sound2))
    silence = AudioSegment.silent(duration=len(sound2) + 100)
    left = silence.overlay(sound, gain_during_overlay=-8)
    right = silence.overlay(sound2, gain_during_overlay=-8)
    stereo_sound = AudioSegment.from_mono_audiosegments(left, right)
    filename = './wav2wavmix/' + audio_file[6:-4] + '_mix_conti_v4.wav'
    stereo_sound.export(filename, format="wav", bitrate="48k")
    print('stereo sound file generated!')


def genMidi(notes,audio_file,initial_tempo = 80,program = 42):
    pm = pmidi.PrettyMIDI(initial_tempo=initial_tempo)
    # print(pm.instruments)
    inst = pmidi.Instrument(program=program, is_drum=False)
    pm.instruments.append(inst)
    # print(pm.instruments)
    velocity = 95
    for i in range(len(notes)):
        note = notes[i]
        pitch = np.int8(np.round(note[2]))
        start_time = note[0] * 128 / 48000
        end_time = note[1] * 128 / 48000
        inst.notes.append(pmidi.Note(velocity, pitch, start_time, end_time))
    pm.write('./wav2midi/'+ audio_file[6:-4] + '.mid')