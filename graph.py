import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from scipy.signal import argrelextrema


SPEED = 0.1


class Graph:

    @staticmethod
    def graph():
        sound, sample_rate = librosa.load('sounds/new_song.wav')

        hl = int(sample_rate * SPEED)
        n_fft = int(sample_rate / 10)
        # Frequencies matrix: each line (frequencies from 0 to the half of the sample)
        # contains data for each time point of the track.
        spec = np.abs(librosa.stft(sound.real, hop_length=hl, n_fft=n_fft))
        # Frequencies corresponded to each column of the spector.
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        # Spector, rotated on 90 degrees.
        rotated_spec = np.rot90(spec)
        # Notes number on each time point.
        n = 1
        # Minimum frequency value.
        min_val = 0
        # Positions of maximum values of each column.
        max_ind = [argrelextrema(rs, np.greater)[0] for rs in rotated_spec]
        max_val_filtered = [
            list(filter(lambda x: x > min_val, np.sort(np.array(rotated_spec[i])[max_ind[i]])[::-1][:n]))
            for i in range(len(rotated_spec))]
        pos = [[x for x in max_ind[i] if rotated_spec[i][x] in max_val_filtered[i]] for i in range(len(rotated_spec))]
        # Average value of each column.
        av_freqs = [sum(freqs[p]) / n for p in pos if len(p) == n][::-1]

        x = [i for i in range(len(av_freqs))]

        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='log')
        plt.subplot(2, 1, 2)
        plt.scatter(x, av_freqs, color='blue', s=3)
        plt.ylim((0, 1200))
        plt.show()


if __name__ == '__main__':
    Graph.graph()
