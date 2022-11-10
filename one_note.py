import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from note import Note


AMPLITUDE = 4096
DURATION = 10
SAMPLE_RATE = 44100


class OneNote:

    @staticmethod
    def plot_note(note: np.ndarray) -> None:
        # Plot the note.
        a = 0
        b = 500

        plt.figure(figsize=(12, 7))
        plt.plot(note[a:b], color='green', linewidth=1)
        plt.show()

    @staticmethod
    def save_as_sound(note: np.ndarray, name: str) -> None:
        # Save the note as a sound.
        wavfile.write(f'sounds/{name}.wav', rate=SAMPLE_RATE, data=note.astype(np.int16))


if __name__ == '__main__':
    freqs = Note.get_freqs()
    note = Note.create(duration=DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE, frequency=freqs['E5'])
    OneNote.plot_note(note)
    OneNote.save_as_sound(note, 'e5')
