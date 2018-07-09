import sys


def cmd_input():
    audio_file = './wav/'
    if len(sys.argv) != 2:
        print('please input a filename,exiting....')
        sys.exit(2)
    else:
        audio_file = audio_file + str(sys.argv[1])
        if audio_file[-4:] != '.wav':
            print('sorry can only deal with wav file now')
            sys.exit(2)
    return audio_file
