import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell

class VanillaLSTMCell(rnn_cell.RNNCell):
    def __init__(self, num_blocks):
        self._num_blocks = num_blocks

    @property
    def input_size(self):
        return self._num_blocks

    @property
    def output_size(self):
        return self._num_blocks

    @property
    def state_size(self):
        return 2 * self._num_blocks

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, y_prev = tf.split(state, 2, 1)

            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_i = get_variable("W_i", [self.input_size, self._num_blocks])
            W_f = get_variable("W_f", [self.input_size, self._num_blocks])
            W_o = get_variable("W_o", [self.input_size, self._num_blocks])

            R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
            R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])
            R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
            R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_i = get_variable("b_i", [1, self._num_blocks])
            b_f = get_variable("b_f", [1, self._num_blocks])
            b_o = get_variable("b_o", [1, self._num_blocks])

            p_i = get_variable("p_i", [self._num_blocks])
            p_f = get_variable("p_f", [self._num_blocks])
            p_o = get_variable("p_o", [self._num_blocks])

            g = h = tf.tanh

            z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.multiply(c_prev, p_i) + b_i)
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f) + tf.multiply(c_prev, p_f) + b_f)
            c = tf.multiply(i, z) + tf.multiply(f, c_prev)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.multiply(c, p_o) + b_o)
            y = tf.multiply(h(c), o)

            return y, tf.concat([c, y],1)
