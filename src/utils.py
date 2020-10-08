import numpy as np
import random
import os
from scipy.io.wavfile import read
from scipy.signal import stft
from PIL import Image, ImageOps
from keras.models import load_model
from keras.models import Model
from keras.layers import Input


def load_audio_dft(path_audio_file, max_feature=64516):
    sr, s1 = read(path_audio_file)
    if len(s1) < max_feature:
        s1 = np.pad(s1, (0, max_feature - len(s1)),
                    mode='constant', constant_values=0)
    else:
        s1 = s1[:max_feature]
    s1 = s1 / (2 ** 10)
    freqs, bins, sxx = stft(s1, nperseg=254 * 2, fs=sr, noverlap=254)
    real_part = np.real(sxx)
    real_part = np.expand_dims(real_part, axis=-1)
    complex_part = np.imag(sxx)
    complex_part = np.expand_dims(complex_part, axis=-1)

    return np.concatenate((real_part, complex_part), axis=-1)


def load_audio_raw(path_audio_file, max_feature=195075):
    sr, s1 = read(path_audio_file)
    if len(s1) < max_feature:
        s1 = np.pad(s1, (0, max_feature - len(s1)),
                    mode='constant', constant_values=0)
    else:
        s1 = s1[:max_feature]
    s1 = np.reshape(s1, (255, 255, 3))
    return s1 / 2 ** 10


def image_generator(path_image_dir, path_audio_dir,
                    batch_size=8, size=(255, 255),
                    preprocess='raw'):
    list_image_train = os.listdir(path_image_dir)
    list_audio_train = os.listdir(path_audio_dir)
    while True:

        batch_image = []
        batch_audio = []

        for i in range(batch_size):
            img_path = os.path.join(path_image_dir,
                                    random.choice(list_image_train))
            audio_path = os.path.join(path_audio_dir,
                                      random.choice(list_audio_train))

            image = Image.open(img_path).convert("RGB")

            image = np.array(ImageOps.fit(image, size),
                             dtype=np.float32) / 255.

            if preprocess == 'raw':
                audio = np.array(load_audio_raw(audio_path), dtype=np.float32)
                batch_image.append(image)
                batch_audio.append(audio)
            elif preprocess == 'dft':
                audio = np.array(load_audio_dft(audio_path), dtype=np.float32)
                batch_image.append(image)
                batch_audio.append(audio)
            else:
                print("Mode Wrong!!!!")

        yield ({'input_image': np.array(batch_image),
                'input_audio': np.array(batch_audio)},
               {'decode_image_model': np.array(batch_image),
                'decode_audio_model': np.array(batch_audio)})


def split_model(path_model='../models/model_dft1.0_1.0.hdf5'):
    model = load_model(path_model)

    model1 = Model(model.inputs, model.outputs[1])

    model1.save("../models/encode_" + path_model.split("/")[-1])

    input_decode = Input(shape=(255, 255, 3))
    output_decode = model.layers[-1](input_decode)

    model2 = Model(input_decode, output_decode)

    model2.save("../models/decode_" + path_model.split("/")[-1])
