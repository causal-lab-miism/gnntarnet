from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout, Conv1D, MaxPool1D
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LocallyConnected1D

tfpl = tfp.layers
tfd = tfp.distributions
class Convolutional1D(Layer):
    def __init__(self, n_c, filters, kernel_size, padding, final_activation, kernel_reg, strides, kernel_init, name, batch_norm=False,  **kwargs):
        super(Convolutional1D, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_c-1):
            self.Layers.append(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=kernel_reg, strides=strides,
                                      kernel_initializer=kernel_init, padding=padding, activation='elu',
                                       name=name+str(i)))
            if batch_norm:
                self.Layers.append(BatchNormalization())
            self.Layers.append(MaxPool1D())
        self.Layers.append(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=kernel_reg, strides=strides,
                                  kernel_initializer=kernel_init, activation=final_activation,
                                      name=name + 'out'))

    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

class FullyConnected(Layer):
    def __init__(self, n_fc, hidden_phi, out_size,  final_activation, name, kernel_reg, kernel_init, activation='elu',
                 bias_initializer=None, dropout=False, batch_norm=False, use_bias=True,  dropout_rate=0.0, **kwargs):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_fc-1):
            if batch_norm:
                self.Layers.append(BatchNormalization())
            self.Layers.append(Dense(units=hidden_phi, activation=activation, kernel_initializer=kernel_init,
                                     bias_initializer=bias_initializer, use_bias=use_bias,
                                     kernel_regularizer=kernel_reg, name=name + str(i)))
            if dropout:
                self.Layers.append(Dropout(dropout_rate))
        self.Layers.append(Dense(units=out_size, activation=final_activation, name=name + 'out'))

    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

class LocallyConnected(Layer):
    def __init__(self, n_layers, filters, out_filters,  final_activation, name, use_bias, kernel_reg, kernel_init, activation='elu',
                 kernel_size=1, batch_norm=False, bias_initializer='zeros', dropout=False, dropout_rate=0.0, **kwargs):
        super(LocallyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_layers-1):
            if batch_norm:
                self.Layers.append(BatchNormalization())
            if dropout:
                self.Layers.append(Dropout(dropout_rate))
            self.Layers.append(LocallyConnected1D(filters=filters, kernel_size=kernel_size, activation=activation,
                                                  kernel_initializer=kernel_init, bias_initializer=bias_initializer,
                                                  kernel_regularizer=kernel_reg, use_bias=use_bias, name=name + str(i), implementation=2))

        self.Layers.append(LocallyConnected1D(filters=out_filters, kernel_size=kernel_size, activation=final_activation,
                                              kernel_initializer=kernel_init, use_bias=use_bias, bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_reg, name=name + 'out', implementation=2))

    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x
class VariationalFullyConnected(Layer):
    def __init__(self, n_fc, hidden_phi, out_size, final_activation, name, kl_weight=1.0, batch_norm=True,
                 activation='elu', **kwargs):
        super(VariationalFullyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_fc - 1):
            if batch_norm:
                self.Layers.append(BatchNormalization())
            self.Layers.append(tfp.layers.DenseVariational(hidden_phi, self.posterior_mean_field, self.prior_trainable,
                                                           kl_weight=1./kl_weight, activation=activation,
                                                           name=name + str(i)))
        if batch_norm:
            self.Layers.append(BatchNormalization())
        self.Layers.append(tfp.layers.DenseVariational(out_size, self.posterior_mean_field, self.prior_trainable,
                                                       kl_weight=1./kl_weight, activation=final_activation,
                                                       name=name + 'out'))
    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return Sequential([
            tfpl.VariableLayer(2 * n, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(n, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1))
        ])
