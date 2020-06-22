import numpy as np
import scipy.io
from tensorflow.keras.utils import to_categorical ,Sequence
from unwrap import unwrap
def normalize_real(x_source):
    a_oo = x_source - x_source.real.min() - 1j * x_source.imag.min()  # origin offsetted
    return a_oo / np.abs(a_oo).max()


def normalize_angle(audio):
    xaudio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
    # def normalize_angle (audio):
    # audio= [item.flatten() for item in audio]
    # audio = min_max_scaler.fit_transform(audio)
    # audio= [item.reshape(256,256) for item in audio]
    return xaudio


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, pair, class_map, batch_size=16, dim=(256, 256, 1), shuffle=True):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            # print (self.pair[i][0])
            img = scipy.io.loadmat(self.pair[i][0])['wrap']
            img_normalized = normalize_angle(img)
            batch_imgs.append(img_normalized)

            label = unwrap(img, wrap_around_axis_0=False, wrap_around_axis_1=False, wrap_around_axis_2=False)
            label_normalized = normalize_angle(label)
            batch_labels.append(label_normalized)

        return np.array(np.expand_dims(batch_imgs, axis=-1)), np.array(np.expand_dims(batch_labels, axis=-1))
