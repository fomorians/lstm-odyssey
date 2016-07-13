from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn.seq2seq import sequence_loss_by_example

# parses the dataset
import ptb_reader

# import variants
from variants.vanilla import VanillaLSTMCell
from variants.nig import NIGLSTMCell
from variants.nfg import NFGLSTMCell
from variants.nog import NOGLSTMCell
from variants.niaf import NIAFLSTMCell
from variants.noaf import NOAFLSTMCell
from variants.np import NPLSTMCell
from variants.cifg import CIFGLSTMCell
from variants.fgr import FGRLSTMCell

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

# load dataset
train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data("ptb")

def write_csv(arr, path):
    df = pd.DataFrame(arr)
    df.to_csv(path)

class PTBModel(object):
    def __init__(self, CellType, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_data")
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="targets")

        lstm_cell = CellType(size)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # initializer used for reusable variable initializer (see `get_variable`)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], initializer=initializer)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        states = []
        state = self.initial_state

        with tf.variable_scope("RNN", initializer=initializer):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                inputs_slice = inputs[:,time_step,:]
                (cell_output, state) = cell(inputs_slice, state)

                outputs.append(cell_output)
                states.append(state)

        self.final_state = states[-1]

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        w = tf.get_variable("softmax_w",
                                    [size, vocab_size],
                                    initializer=initializer)
        b = tf.get_variable("softmax_b", [vocab_size], initializer=initializer)

        logits = tf.nn.xw_plus_b(output, w, b) # compute logits for loss
        targets = tf.reshape(self.targets, [-1]) # reshape our target outputs
        weights = tf.ones([batch_size * num_steps]) # used to scale the loss average

        # computes loss and performs softmax on our fully-connected output layer
        loss = sequence_loss_by_example([logits], [targets], [weights], vocab_size)
        self.cost = cost = tf.div(tf.reduce_sum(loss), batch_size, name="cost")

        if is_training:
            # setup learning rate variable to decay
            self.lr = tf.Variable(1.0, trainable=False)

            # define training operation and clip the gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), name="train")
        else:
            # if this model isn't for training (i.e. testing/validation) then we don't do anything here
            self.train_op = tf.no_op()

def run_epoch(sess, model, data, verbose=False):
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()

    # accumulated counts
    costs = 0.0
    iters = 0

    # initial RNN state
    state = model.initial_state.eval()

    for step, (x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        cost, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict={
            model.input_data: x,
            model.targets: y,
            model.initial_state: state
        })
        costs += cost
        iters += model.num_steps

        perplexity = np.exp(costs / iters)

        if verbose and step % 10 == 0:
            progress = (step / epoch_size) * 100
            wps = iters * model.batch_size / (time.time() - start_time)
            print("%.1f%% Perplexity: %.3f (Cost: %.3f) Speed: %.0f wps" % (progress, perplexity, cost, wps))

    return (costs / iters), perplexity

class Config(object):
    batch_size = 20
    num_steps = 35 # number of unrolled time steps
    hidden_size = 450 # number of blocks in an LSTM cell
    vocab_size = 10000
    max_grad_norm = 5 # maximum gradient for clipping
    init_scale = 0.05 # scale between -0.1 and 0.1 for all random initialization
    keep_prob = 0.5 # dropout probability
    num_layers = 2 # number of LSTM layers
    learning_rate = 1.0
    lr_decay = 0.8
    lr_decay_epoch_offset = 6 # don't decay until after the Nth epoch

# default settings for training
train_config = Config()

# our evaluation runs (validation and testing), use a batch size and time step of one
eval_config = Config()
eval_config.batch_size = 1
eval_config.num_steps = 1

# number of epochs to perform over the training data
num_epochs = 39

cell_types = {
    'vanilla': VanillaLSTMCell,
    'nig': NIGLSTMCell,
    'nfg': NFGLSTMCell,
    'nog': NOGLSTMCell,
    'niaf': NIAFLSTMCell,
    'noaf': NOAFLSTMCell,
    'np': NPLSTMCell,
    'cifg': CIFGLSTMCell,
    'fgr': FGRLSTMCell,
}

model_name = "vanilla"
CellType = cell_types[model_name]

with tf.Graph().as_default(), tf.Session() as sess:
    # define our training model
    with tf.variable_scope("model", reuse=None):
        train_model = PTBModel(CellType, is_training=True, config=train_config)

    # we create a separate model for validation and testing to alter the batch size and time steps
    # reuse=True reuses variables from the previously defined `train_model`
    with tf.variable_scope("model", reuse=True):
        valid_model = PTBModel(CellType, is_training=False, config=train_config)
        test_model = PTBModel(CellType, is_training=False, config=eval_config)

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, '%s.pb'.format(model_name), as_text=False)

    train_costs = []
    train_perps = []
    valid_costs = []
    valid_perps = []

    for i in range(num_epochs):
        print("Epoch: %d Learning Rate: %.3f" % (i + 1, sess.run(train_model.lr)))

        # run training pass
        train_cost, train_perp = run_epoch(sess, train_model, train_data, verbose=True)
        print("Epoch: %i Training Perplexity: %.3f (Cost: %.3f)" % (i + 1, train_perp, train_cost))
        train_costs.append(train_cost)
        train_perps.append(train_perp)

        # run validation pass
        valid_cost, valid_perp = run_epoch(sess, valid_model, valid_data)
        print("Epoch: %i Validation Perplexity: %.3f (Cost: %.3f)" % (i + 1, valid_perp, valid_cost))
        valid_costs.append(valid_cost)
        valid_perps.append(valid_perp)

        saver.save(sess, checkpoint_path + 'checkpoint')

    # run test pass
    test_cost, test_perp = run_epoch(sess, test_model, test_data)
    print("Test Perplexity: %.3f (Cost: %.3f)" % (test_perp, test_cost))

    write_csv(train_costs, os.path.join(summary_path, "train_costs.csv"))
    write_csv(train_perps, os.path.join(summary_path, "train_perps.csv"))
    write_csv(valid_costs, os.path.join(summary_path, "valid_costs.csv"))
    write_csv(valid_perps, os.path.join(summary_path, "valid_perps.csv"))
