import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class FGRLSTMCell(rnn_cell.RNNCell):
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
        return 5 * self._num_blocks

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, y_prev, i_prev, f_prev, o_prev = tf.split(1, 5, state)

            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_i = get_variable("W_i", [self.input_size, self._num_blocks])
            W_f = get_variable("W_f", [self.input_size, self._num_blocks])
            W_o = get_variable("W_o", [self.input_size, self._num_blocks])

            R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
            R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])
            R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
            R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

            R_ii = get_variable("R_ii", [self._num_blocks, self._num_blocks])
            R_fi = get_variable("R_fi", [self._num_blocks, self._num_blocks])
            R_oi = get_variable("R_oi", [self._num_blocks, self._num_blocks])

            R_if = get_variable("R_if", [self._num_blocks, self._num_blocks])
            R_ff = get_variable("R_ff", [self._num_blocks, self._num_blocks])
            R_of = get_variable("R_of", [self._num_blocks, self._num_blocks])

            R_io = get_variable("R_io", [self._num_blocks, self._num_blocks])
            R_fo = get_variable("R_fo", [self._num_blocks, self._num_blocks])
            R_oo = get_variable("R_oo", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_i = get_variable("b_i", [1, self._num_blocks])
            b_f = get_variable("b_f", [1, self._num_blocks])
            b_o = get_variable("b_o", [1, self._num_blocks])

            p_i = get_variable("p_i", [self._num_blocks])
            p_f = get_variable("p_f", [self._num_blocks])
            p_o = get_variable("p_o", [self._num_blocks])

            g = h = tf.tanh

            z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)

            i_bar = tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.mul(c_prev, p_i) + b_i + tf.matmul(i_prev, R_ii) + tf.matmul(f_prev, R_fi) + tf.matmul(o_prev, R_oi)
            i = tf.sigmoid(i_bar)

            f_bar = tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f) + tf.mul(c_prev, p_f) + b_f + tf.matmul(i_prev, R_if) + tf.matmul(f_prev, R_ff) + tf.matmul(o_prev, R_of)
            f = tf.sigmoid(f_bar)

            c = tf.mul(i, z) + tf.mul(f, c_prev)

            o_bar = tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.mul(c, p_o) + b_o + tf.matmul(i_prev, R_io) + tf.matmul(f_prev, R_fo) + tf.matmul(o_prev, R_oo)
            o = tf.sigmoid(o_bar)

            y = tf.mul(h(c), o)

            return y, tf.concat(1, [c, y, i, f, o])
