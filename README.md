# Melody Extraction


# Usage
first drag your .wav file into folder /wav

- for left channel melody float frequency and right channel vocal
```bash
>python wav2wav.py infile.wav
```

- for generate midi file
```bash
>python wav2midi.py infile.wav
```

# Dependencies:
- Anaconda: Alternative , but recommended
- Librosa: https://github.com/librosa/librosa (pip install librosa)
- NumPy & SciPy: http://www.scipy.org/ 
- pydub(used for wav2wav.py): https://github.com/jiaaro/pydub (pip install pydub) 
- pretty_midi(used for wav2midi.py): https://github.com/craffel/pretty-midi (pip install pretty_midi)