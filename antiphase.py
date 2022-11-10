import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

from note import Note
from one_note import OneNote


AMPLITUDE = 4096
DURATION = 10
SAMPLE_RATE = 44100


class Antiphase:

    @staticmethod
    def anti_note(note: np.ndarray) -> np.ndarray:
        # Create the opposite version of the note.
        return note * -1

    @staticmethod
    def plot_notes(note_1: np.ndarray, note_2: np.ndarray) -> None:
        # Plot the notes.
        a = 0
        b = 500

        plt.figure(figsize=(12, 7))
        plt.plot(note_1[a:b], color='green', linewidth=1)
        plt.plot(note_2[a:b], color='red', linewidth=1)
        plt.xlim((a, b))
        plt.grid()
        plt.show()

    @staticmethod
    def mix(note_1: str, note_2: str) -> None:
        # Combine notes with opposite data.
        audio1 = AudioSegment.from_file(f'sounds/{note_1}.wav')
        audio2 = AudioSegment.from_file(f'sounds/{note_2}.wav')
        mixed = audio1.overlay(audio2)
        mixed.export('sounds/silence.wav', format='wav')


if __name__ == '__main__':
    # Create two note with opposite data. The result is silence.
    freqs = Note.get_freqs()
    note_1 = Note.create(duration=DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE, frequency=freqs['E5'])
    note_2 = Antiphase.anti_note(note_1)
    OneNote.save_as_sound(note_2, 'e5_anti')

    Antiphase.plot_notes(note_1, note_2)
    Antiphase.mix('e5', 'e5_anti')
