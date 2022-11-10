import librosa


class Compressor:

    @staticmethod
    def compress():
        sound, sample_rate = librosa.load('sounds/new_song.wav')
        print(sound)


if __name__ == '__main__':
    Compressor.compress()
