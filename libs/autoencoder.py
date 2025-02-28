import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Dense, Flatten, UpSampling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from typing import List


import numpy as np
import matplotlib.pyplot as plt

# we will define the encoder and decoder as models using the functional API.
class Autoencoder:
    def __init__(self, input_shape: tuple, conv_filters: List[int],
                 conv_kernels: List[int], conv_strides: List[str], 
                 latent_dim: int):
        
        self.input_shape = input_shape # e.g. (28, 28, 1)
        self.conv_filters = conv_filters # e.g. [32, 64, 128, 256]
        self.conv_kernels = conv_kernels # e.g. [3, 3, 3, 3]
        self.conv_strides = conv_strides # e.g. [1, 1, 1, 1], stride 2 for downsampling
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)

        self._shape_before_bottleneck = None
        self._model_input = None

        self._build() # uses the build method 

    def _build(self):
        # build the encoder
        self._build_encoder()
        # build the decoder
        self._build_decoder()
        self.build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input) # input gets passed to conv layers
        bottleneck = self._add_bottleneck(conv_layers) # output after conv layers gets passed to bottleneck
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder") # Model object constructs graph define by tensor flow
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_upsample_layers = self._add_conv_upsample_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_upsample_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    
    def build_autoencoder(self):
        model_input = self._model_input
        model_output = self._decoder(self._encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """Creates convolutional blocks for the encoder"""
        x = encoder_input 
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(i, x) 
        return x
    
    def _add_conv_layer(self, layer_idx, x):
        """Adds a convolutional block to the encoder graph of layers.
        Conv layer, ReLU, BatchNorm
        """
        conv_layer = Conv2D(filters=self.conv_filters[layer_idx], 
                            kernel_size=self.conv_kernels[layer_idx], 
                            strides=self.conv_strides[layer_idx], 
                            padding="same", 
                            name=f"encoder_conv_layer_{layer_idx+1}")
        x = conv_layer(x)
        x = BatchNormalization(name=f"encoder_batchnorm_{layer_idx+1}")(x)
        x = ReLU(name=f"encoder_relu_{layer_idx+1}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten the output of the last conv layer and feed it into a dense layer."""
        # shape of x is (batch_size, height, width, channels)
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_dim, name="encoder_output")(x)
        return x
    
    def _add_decoder_input(self):
        return Input(shape=(self.latent_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # product of the shape of the output of the last conv layer
        dense_layer = Dense(num_neurons, name="dense_layer")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        # reshape layer applid to dense layer output
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer


    def _add_conv_upsample_layers(self, x):
        """Adds conv transpose blocks for the decoder"""
        for layer_idx in reversed(range(1, self._num_conv_layers)):
            if self.conv_strides[layer_idx] > 1:
                # Only upsample when there was a stride > 1 in encoder
                x = UpSampling2D(size=self.conv_strides[layer_idx])(x)
            x = self._add_conv_layer_decoder(layer_idx, x)
        return x

    def _add_conv_layer_decoder(self, layer_idx, x):

        """Adds a conv block to the decoder graph of layers."""
        layer_num = self._num_conv_layers - layer_idx
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_idx],
            kernel_size=self.conv_kernels[layer_idx],
            padding="same",
            name=f"decoder_conv_layer_{layer_num}"
        )(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(conv_layer)
        x = BatchNormalization(name=f"decoder_batchnorm_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        """Adds the final conv block to the decoder graph of layers."""
        if self.conv_strides[0] > 1:
            x = UpSampling2D(size=self.conv_strides[0])(x)
        x = Conv2D(
            filters=1,  # or input_shape[-1] for flexible channel count
            kernel_size=self.conv_kernels[0],
            padding="same",
            activation='sigmoid',  # typically used for image reconstruction
            name=f"decoder_conv_layer_{self._num_conv_layers}"
        )(x)
        return x
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size=32, epochs=10):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True)
