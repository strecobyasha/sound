import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import argrelextrema

from note import Note


AMPLITUDE = 4096
DURATION = 1
SAMPLE_RATE = 44100


class Inverse:

    @staticmethod
    def create_sound() -> np.ndarray:
        # Create a mix of two notes.
        freqs = Note.get_freqs()
        note_1 = Note.create(duration=DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE, frequency=freqs['E5'])
        note_2 = Note.create(duration=DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE, frequency=freqs['A5'])
        sound = np.sum([note_1, note_2], axis=0)

        return sound

    @staticmethod
    def save_as_sound(note: np.ndarray, name: str) -> None:
        # Save the mix as a sound.
        wavfile.write(f'sounds/{name}.wav', rate=SAMPLE_RATE, data=note.astype(np.int16))

    @staticmethod
    def plot_sound(sound: np.ndarray) -> None:
        # Plot a sound data.
        a = 0
        b = 500
        plt.figure(figsize=(12, 7))
        plt.plot(sound[a:b], color='green', linewidth=1)
        plt.xlim((a, b))
        plt.show()

    @staticmethod
    def inverse_sound(sound: np.ndarray) -> np.ndarray:
        # Inversing sound.
        return np.fft.fft(sound)

    @staticmethod
    def plot_inversed_sound(sound: np.ndarray) -> None:
        # Since the sound contains two notes, on the plot will be two picks corresponding to those two notes.
        a = 0
        b = 1000
        plt.figure(figsize=(12, 7))
        plt.plot(sound[a:b], color='green', linewidth=1)
        plt.xlim((a, b))
        plt.show()

    @staticmethod
    def get_notes_from_inversed(sound: np.ndarray) -> None:
        # X-coordinates of the picks in the plot are corresponded with notes frequencies.
        max_ind = argrelextrema(sound[:int(SAMPLE_RATE / 2)], np.greater)[0]
        print(max_ind)


if __name__ == '__main__':
    sound = Inverse.create_sound()
    Inverse.save_as_sound(sound, 'e5_a5')
    Inverse.plot_sound(sound)
    inversed_sound = Inverse.inverse_sound(sound)
    Inverse.plot_inversed_sound(inversed_sound)
    Inverse.get_notes_from_inversed(inversed_sound)
