import numpy as np
import tensorflow as tf

class Autoencoder:

    def __init__(self, input = None, input_shape = None):
        self.input = input
        self.input_shape = input_shape

        self._define_network()

    def _define_encoder(self, current):
        pass

    def _define_decoder(self, current):
        pass

    def _define_network(self):

        if self.input is None:
            self.input = tf.placeholder(dtype= tf.float32, shape= self.input_shape, name= 'input')

        with tf.variable_scope('encoder'):
            self.latent_z = self._define_encoder(self.input)

        with tf.variable_scope('decoder'):
            self.reconstruction = self._define_decoder(self.latent_z)

        self._define_loss()

    def _define_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.input, self.reconstruction))


class VariationalAutoencoder(Autoencoder):

    def __init__(self, input = None, input_shape = None):
        super().__init__(input, input_shape)

    def _define_network(self):

        if self.input is None:
            self.input = tf.placeholder(dtype= tf.float32, shape= self.input_shape, name= 'input')

        with tf.variable_scope('encoder'):
            self.z_mu, self.z_log_sigma = self._define_encoder(self.input)

            random_factor = tf.distributions.Normal(loc= 0.0, scale= 1.0)
            self.latent_z = self.z_mu + tf.exp(self.z_log_sigma / 2) * random_factor.sample(sample_shape= tf.shape(self.z_mu))

        with tf.variable_scope('decoder'):
            self.reconstruction = self._define_decoder(self.latent_z)

        self._define_loss()

    def _define_loss(self):
        # Define the loss
        self.recon_loss = tf.reduce_sum(tf.square(self.input - self.reconstruction),reduction_indices= 1)

        self.kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma - tf.square(self.z_mu) - tf.exp(self.z_log_sigma),
                                            reduction_indices=1)

        # Take the mean over the batch
        self.loss = tf.reduce_mean(self.recon_loss + self.kl_loss)

