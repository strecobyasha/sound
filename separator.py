import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

import librosa

import librosa.display
from scipy.io import wavfile


class Separator:

    @staticmethod
    def vocal_sep():
        # sound, sample_rate = librosa.load(librosa.ex('fishin'), duration=120)
        sound, sample_rate = librosa.load("sounds/Karissa_Hobbs_-_Let's_Go_Fishin'.ogg")
        # And compute the spectrogram magnitude and phase
        S_full, phase = librosa.magphase(librosa.stft(sound))
        idx = slice(*librosa.time_to_frames([10, 15], sr=sample_rate))
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
            y_axis='log',
            x_axis='time',
            sr=sample_rate, ax=ax,
        )
        fig.colorbar(img, ax=ax)
        # plt.show()

        # We'll compare frames using cosine similarity, and aggregate similar frames
        # by taking their (per-frequency) median value.
        #
        # To avoid being biased by local continuity, we constrain similar frames to be
        # separated by at least 2 seconds.
        #
        # This suppresses sparse/non-repetetitive deviations from the average spectrum,
        # and works well to discard vocal elements.
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sample_rate)))

        # The output of the filter shouldn't be greater than the input
        # if we assume signals are additive.  Taking the pointwise minimum
        # with the input spectrum forces this.
        S_filter = np.minimum(S_full, S_filter)
        # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
        # Note: the margins need not be equal for foreground and background separation
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        # Once we have the masks, simply multiply them with the input spectrum
        # to separate the components
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # sphinx_gallery_thumbnail_number = 2
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
        img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                                       y_axis='log', x_axis='time', sr=sample_rate, ax=ax[0])
        ax[0].set(title='Full spectrum')
        ax[0].label_outer()

        librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sample_rate, ax=ax[1])
        ax[1].set(title='Background')
        ax[1].label_outer()

        librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sample_rate, ax=ax[2])
        ax[2].set(title='Foreground')
        fig.colorbar(img, ax=ax)

        # plt.show()

        y_foreground = librosa.istft(S_foreground * phase) * (-10000)
        wavfile.write(
            f'sounds/vocal.wav',
            rate=sample_rate,
            data=y_foreground[10*sample_rate:15*sample_rate].astype(np.int16),
        )

        y_background = librosa.istft(S_background * phase) * (-10000)
        wavfile.write(
            f'sounds/back.wav',
            rate=sample_rate,
            data=y_background[10*sample_rate:15 * sample_rate].astype(np.int16),
        )


if __name__ == '__main__':
    Separator.vocal_sep()
