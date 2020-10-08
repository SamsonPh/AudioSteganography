from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model


def BaseModel(input_tensor):
    x3 = Conv2D(50, activation='relu',
                kernel_size=3, padding="same")(input_tensor)
    x3 = Conv2D(50, activation='relu',
                kernel_size=3, padding="same")(x3)
    x3 = Conv2D(50, activation='relu',
                kernel_size=3, padding="same")(x3)
    x3 = Conv2D(50, activation='relu',
                kernel_size=3, padding="same")(x3)
    x3 = Conv2D(50, activation='relu',
                kernel_size=3, padding="same")(x3)

    x4 = Conv2D(50, activation='relu',
                kernel_size=4, padding="same")(input_tensor)
    x4 = Conv2D(50, activation='relu',
                kernel_size=4, padding="same")(x4)
    x4 = Conv2D(50, activation='relu',
                kernel_size=4, padding="same")(x4)
    x4 = Conv2D(50, activation='relu',
                kernel_size=4, padding="same")(x4)
    x4 = Conv2D(50, activation='relu',
                kernel_size=4, padding="same")(x4)

    x5 = Conv2D(50, activation='relu',
                kernel_size=5, padding="same")(input_tensor)
    x5 = Conv2D(50, activation='relu',
                kernel_size=5, padding="same")(x5)
    x5 = Conv2D(50, activation='relu',
                kernel_size=5, padding="same")(x5)
    x5 = Conv2D(50, activation='relu',
                kernel_size=5, padding="same")(x5)
    x5 = Conv2D(50, activation='relu',
                kernel_size=5, padding="same")(x5)

    x = Concatenate(axis=3)([x3, x4, x5])

    return x


def Decode_Image_Model(channels):
    input_audio = Input(shape=(255, 255, channels), name='input_audio')
    input_image = Input(shape=(255, 255, 3), name='input_image')

    encoded_audio = BaseModel(input_audio)
    concat_input = Concatenate(axis=3)([encoded_audio, input_image])

    decoded_image = BaseModel(concat_input)
    decoded_image = Conv2D(3, activation='relu',
                           kernel_size=1, padding="same",
                           name="decode_image")(decoded_image)

    return Model(inputs=[input_audio, input_image],
                 outputs=[decoded_image], name='decode_image_model')


def Decode_Audio_Model(channels):
    decoded_image = Input(shape=(255, 255, 3))

    decoded_audio = BaseModel(decoded_image)
    decoded_audio = Conv2D(channels, activation='linear',
                           kernel_size=1, padding="same",
                           name="decode_audio")(decoded_audio)

    return Model(input=decoded_image,
                 output=decoded_audio,
                 name='decode_audio_model')


def Encode_Decoder_Model(preprocess="dft"):
    if preprocess == "dft":
        channels = 2
    else:
        channels = 3

    decode_image_model = Decode_Image_Model(channels)
    decode_audio_model = Decode_Audio_Model(channels)

    input_audio = Input(shape=(255, 255, channels), name="input_audio")
    input_image = Input(shape=(255, 255, 3), name="input_image")

    decode_image = decode_image_model([input_audio, input_image])
    decode_audio = decode_audio_model(decode_image)

    return Model(inputs=[input_audio, input_image],
                 outputs=[decode_audio, decode_image])


# encode_decoder_model = Encode_Decoder_Model("raw")
# encode_decoder_model.summary()
