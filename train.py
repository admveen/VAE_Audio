from libs.var_autoencoder import VarAutoencoder
from tensorflow.keras.datasets import mnist
import numpy as np

# global training parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize the data
    x_train = x_train.astype("float") / 255
    x_train = x_train[:,:,:, np.newaxis]
    x_train = x_train.astype("float") / 255
    x_test = x_test[:,:,:, np.newaxis]

    return x_train, y_train, x_test, y_test

def train(x_train, lr, batch_size, epochs):
    vae_mod = VarAutoencoder(
        input_shape = (28,28,1),
        conv_filters = (32, 64,64,64),
        conv_kernels = (3,3,3,3),
        conv_strides = (1,2,2,1),
        latent_space_dim = 2)
    vae_mod.summary()
    vae_mod.compile(learning_rate = lr)
    vae_mod.train(x_train, batch_size=batch_size, epochs = epochs)

    return vae_mod

if __name__ == "__main__":

    x_train, _, _, _ = load_mnist()
    trained_vae = train(x_train[0:500], LEARNING_RATE, BATCH_SIZE, EPOCHS)
