import argparse
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from model import Encode_Decoder_Model
from utils import image_generator

PATH_IMAGE_TRAIN = "../train/image"
PATH_AUDIO_TRAIN = "../train/audio"
PATH_IMAGE_VAL = "../validate/image"
PATH_AUDIO_VAL = "../validate/audio"

def train(args):

    audio_fraction = args.audio_fraction
    image_fraction = args.image_fraction
    preprocess = args.preprocess

    encode_decoder_model = Encode_Decoder_Model(preprocess)
    encode_decoder_model.summary()
    encode_decoder_model.compile(optimizer=Adam(lr=0.0001),
                                 loss={'decode_image_model': 'mse', 'decode_audio_model': 'mse'},
                                 loss_weights={'decode_image_model': image_fraction, 'decode_audio_model': audio_fraction})

    check_point = ModelCheckpoint('../models/model_' + preprocess + str(image_fraction) + '_' + str(audio_fraction) + '.hdf5',
                                  verbose=True, save_best_only=True)
    early_stop = EarlyStopping(patience=5, verbose=True)
    encode_decoder_model.fit_generator(image_generator(PATH_IMAGE_TRAIN, PATH_AUDIO_TRAIN, mode=preprocess, batch_size=8),
                                       steps_per_epoch=500, epochs=100,
                                       validation_data=image_generator(PATH_IMAGE_VAL, PATH_AUDIO_VAL, mode=preprocess),
                                       validation_steps=200, callbacks=[check_point, early_stop])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_fraction', type=float, help='audio_fraction', default=1.)
    parser.add_argument('--image_fraction', type=float, help='image_fraction', default=1.)
    parser.add_argument('--preprocess', type=str, help='preprocess: raw/dft, default:dft',
                        default="dft", choices=["dft", "raw"])
    return parser.parse_args(argv)


if __name__ == '__main__':
    train(parse_arguments(sys.argv[1:]))