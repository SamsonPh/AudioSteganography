import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from scipy.io.wavfile import write
from scipy.signal import istft
from utils import load_audio_dft, load_audio_raw


def decode_image_dft(args):
    path_model = args.path_model
    img_path = args.img_path
    audio_path = args.audio_path

    image = Image.open(img_path).convert("RGB")
    audio = np.array(load_audio_dft(audio_path), dtype=np.float32)

    image = np.array(ImageOps.fit(image, (255, 255)), dtype=np.float32) / 255.

    image = np.expand_dims(image, axis=0)
    audio = np.expand_dims(audio, axis=0)

    decode_image_model = load_model(path_model)
    image_decode = decode_image_model.predict({'input_image': image, 'input_audio': audio})

    image_decode = (np.squeeze(image_decode) * 255).astype("int")

    plt.imsave("../outputs/image_decode.png", image_decode)


def decode_audio_dft(args):
    path_model = args.path_model
    img_path = args.img_path

    decode_model = load_model(path_model)

    test = Image.open(img_path).convert("RGB")
    test = np.array(ImageOps.fit(test, (255, 255)), dtype=np.float32) / 255.

    audio = np.squeeze(decode_model.predict(np.expand_dims(test, axis=0)))

    audio *= (2 ** 10)
    audio = audio[:, :, 0] + audio[:, :, 1] * 1j

    t, x = istft(np.abs(audio) * np.exp(1j * np.angle(audio)), fs=16000, nfft=254 * 2, nperseg=254 * 2, noverlap=254)
    x = x.astype(np.int16)

    write("../outputs/audio_decode.wav", 16000, x)

    return x


def decode_image_raw(args):
    path_model = args.path_model
    img_path = args.img_path
    audio_path = args.audio_path

    image = Image.open(img_path).convert("RGB")
    audio = np.array(load_audio_raw(audio_path), dtype=np.float32)

    image = np.array(ImageOps.fit(image, (255, 255)), dtype=np.float32) / 255.

    image = np.expand_dims(image, axis=0)
    audio = np.expand_dims(audio, axis=0)

    decode_image_model = load_model(path_model)
    image_decode = decode_image_model.predict({'input_image': image, 'input_audio': audio})

    image_decode = (np.squeeze(image_decode) * 255).astype("int")

    plt.imsave("../outputs/image_decode.png", image_decode)


def decode_audio_raw(args):
    path_model = args.path_model
    img_path = args.img_path

    decode_model = load_model(path_model)

    test = Image.open(img_path).convert("RGB")
    test = np.array(ImageOps.fit(test, (255, 255)), dtype=np.float32) / 255.

    audio = np.squeeze(decode_model.predict(np.expand_dims(test, axis=0)))

    audio *= (2 ** 10)
    audio = np.reshape(audio, 255 ** 2 * 3)
    audio = audio.astype(np.int16)

    write("../outputs/audio_decode.wav", 16000, audio)

    return audio


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='model using: e for encoding, d for decoding(d)', choices=["e", "d"])
    parser.add_argument('--preprocess', type=str, help='preprocess: raw/dft, default:dft',
                        default="dft", choices=["dft", "raw"])
    parser.add_argument('--path_model', type=str, help='path to model pre-trained')
    parser.add_argument('--img_path', type=str, help='path to image')
    parser.add_argument('--audio_path', type=str, help='path to audio')
    return parser.parse_args(argv)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    if arg.mode == "e":
        if arg.preprocess == "dft":
            decode_image_dft(arg)
        if arg.preprocess == "raw":
            decode_image_raw(arg)
    elif arg.mode == "d":
        if arg.preprocess == "dft":
            decode_audio_dft(arg)
        if arg.preprocess == "raw":
            decode_audio_raw(arg)
    else:
        print("Error")
