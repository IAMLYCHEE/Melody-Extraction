import sys


def cmd_input():
    audio_file = './wav/'
    if len(sys.argv) < 2:
        print('please input a filename,exiting....')
        sys.exit(2)
    else:
        audio_file = audio_file + str(sys.argv[1])
        if audio_file[-4:] != '.wav':
            print('sorry can only deal with wav file now')
            sys.exit(2)
        if sys.argv == 3:
            mode = str(sys.argv[2])
            if mode == 'wav2wav':
                print('check ./wav2wavmix folder')
            elif mode == 'wav2midi':
                print('check ./wav2midi folder')
            else:
                print('default mode, check ./wav2wavmix  and ./wav2midi folder')
        if sys.argv != 3:
            print('default mode, check ./wav2wavmix  and ./wav2midi folder')
            mode = 'all'

    return audio_file,mode
