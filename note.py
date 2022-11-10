import numpy as np


class Note:

    @staticmethod
    def get_freqs() -> dict:
        # Create a dictionary of 88 notes from A0 to C8, where key is a note
        # and value is frequency.
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
        base_freq = 440
        keys = np.array([n + str(i) for i in range(0, 9) for n in octave])
        keys = keys[np.where(keys == 'A0')[0][0]:(np.where(keys == 'C8')[0][0] + 1)]
        note_freqs = dict(zip(keys, [base_freq * 2 ** ((n + 1 - 49) / 12) for n in range(len(keys))]))
        note_freqs[''] = 0.0

        return note_freqs

    @staticmethod
    def create(duration: float, sample_rate: int, amplitude: int, frequency: int) -> np.ndarray:
        # Create a note.
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = amplitude * np.sin(2 * np.pi * frequency * t)

        return wave
