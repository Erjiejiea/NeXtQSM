import tensorflow as tf

from network.unet import UNet
from utils import misc


class VarNet(tf.keras.Model):

    def __init__(self, config):
        super(VarNet, self).__init__()
        self.vn_n_layers = config.vn_n_layers
        self.vn_starting_filters = config.vn_starting_filters
        self.vn_kernel_initializer = config.vn_kernel_initializer
        self.vn_batch_norm = config.vn_batch_norm
        self.vn_act_func = config.vn_act_func

        self.vn_n_steps = config.vn_n_steps
        self.vn_l_init = config.vn_l_init

        self.nets, self.lambdas = self.init_layers()

    def init_layers(self):
        lambdas = []

        nets = UNet(1, self.vn_n_layers, self.vn_starting_filters, 3, self.vn_kernel_initializer, self.vn_batch_norm,
                    0., misc.get_act_function(self.vn_act_func), 1, False, False, None)

        for n_step in range(self.vn_n_steps):
            lambdas.append(tf.Variable(self.vn_l_init, name='L-' + str(n_step), trainable=True, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0., 500)))

        return nets, lambdas

    def call(self, x, training):
        return self.nets(x, training=training)

    def summary(self, input_shape):
        """
        :param input_shape: (32, 32, 1)  my input_shape: (256, 256, 256, 1)
        """
        x = tf.keras.Input(shape=input_shape)
        print(input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))  # False
        model.summary(line_length=130)
