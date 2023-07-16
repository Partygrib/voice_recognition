from keras.layers import *
from keras.callbacks import *


class channel_attention(Layer):

    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(channel // self.ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = GlobalAveragePooling3D()(inputs)
        avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling3D()(inputs)
        max_pool = Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = Add()([avg_pool, max_pool])
        feature = Activation('sigmoid')(feature)

        return multiply([inputs, feature])


class spatial_attention(Layer):

    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv3d = Conv3D(filters=1, kernel_size=self.kernel_size,
                             strides=1, padding='same', activation='sigmoid',
                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = Lambda(lambda x: backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)

        return multiply([inputs, feature])


def cbam_block(feature, ratio=512, kernel_size=(3, 3, 3)):
    feature = channel_attention(ratio=ratio)(feature)
    feature = spatial_attention(kernel_size=kernel_size)(feature)

    return feature
