# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import copy
from sklearn.metrics import mean_absolute_error


def read_csv_file(filename):
    """
    read csv file
    :param filename:
    :return:
    """
    data = pd.read_csv(filename, sep=',', header=0)
    return data


# 读数据
df = read_csv_file('data/all.csv')
# 取其中的有用项
input_data = df.loc[:, ['StartPrice', 'EndPrice', 'HighPrice', 'LowPrice', 'Total_ContNum', 'AvePrice']].values
output_data = df.loc[:, ['AvePrice']].values

# 绘制均价走势
# plt.plot(output_data, label="Ave")
# plt.legend()
# plt.show()

# 归一化
min_max_scaler = MinMaxScaler()
input_data = min_max_scaler.fit_transform(input_data)
scaler_for_output = MinMaxScaler(feature_range=(0, 1))
output_data = scaler_for_output.fit_transform(output_data)#scaler_for_output将被用于还原
# 窗口长度
window_len = 21
# 测试集长度
test_len = 10
# 划分train和test
normalized_train_data = input_data[0:-(2*window_len+test_len)]
normalized_test_data = input_data[-(2*window_len+test_len):-window_len]
label_train = output_data[window_len:-(window_len+test_len)]
label_test = output_data[-(window_len+test_len):]

# 训练集
train_x = []
for i in range(0, len(normalized_train_data)-window_len):
    temp_set = normalized_train_data[i:(i + window_len)].copy()
    train_x.append(temp_set)
train_x = np.array(train_x)
train_y = []
for i in range(0, len(label_train)-window_len):
    temp_set = label_train[i:(i + window_len)].copy()
    train_y.append(temp_set)
train_y = np.array(train_y)
# 测试集
test_x = []
for i in range(0, len(normalized_test_data) - window_len):
    temp_set = normalized_test_data[i:(i + window_len)].copy()
    test_x.append(temp_set)
test_x = np.array(test_x)
test_y = []
for i in range(0, len(label_test)-window_len):
    temp_set = label_test[i:(i + window_len)].copy()
    test_y.append(temp_set)
test_y = np.array(test_y)

print("Train & Test Dataset Shape: ")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
#print(test_x[:, 0])
# model
## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003

## Network Parameters
# length of input signals
input_seq_len = 21
# length of output signals
output_seq_len = 21
# size of LSTM Cell
hidden_dim = 32
# num of input signals
input_dim = 6
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5


def build_graph(feed_previous=False):
    tf.reset_default_graph()

    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    weights = {
        'out': tf.get_variable('Weights_out',
                               shape=[hidden_dim, output_dim],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out',
                               shape=[output_dim],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
          """RNN decoder for the sequence-to-sequence model.
          Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            loop_function: If not None, this function will be applied to the i-th output
              in order to generate the i+1-st input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
              state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
          """
          with tf.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                tf.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=tf.float32,
                              scope=None):
          """Basic RNN sequence-to-sequence model.
          This model first runs an RNN to encode encoder_inputs into a state vector,
          then runs decoder, initialized with the last encoder state, on decoder_inputs.
          Encoder and decoder use the same RNN cell type, but don't share parameters.
          Args:
            encoder_inputs: A list of 2D Tensors [batch_size x input_size].
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), all other inputs will be generated by the previous
              decoder output using _loop_function below. If False, decoder_inputs are used
              as given (the standard decoder case).
            dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          with tf.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = tf.contrib.rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        enc_inp=enc_inp,
        target_seq=target_seq,
        train_op=optimizer,
        loss=loss,
        saver=saver,
        reshaped_outputs=reshaped_outputs,
        )


total_iteractions = 7000
#batch_size = 32
KEEP_RATE = 0.5
train_losses = []
val_losses = []


rnn_model = build_graph(feed_previous=False)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    for i in range(total_iteractions):
        batch_input = train_x
        batch_output = train_y

        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t].reshape(-1,input_dim) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./model/canshu/', 'univariate_ts_mode21_7K_0'))

print("Checkpoint saved at: ", save_path)


rnn_model = build_graph(feed_previous=True)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    saver = rnn_model['saver']().restore(sess, os.path.join('./model/canshu/', 'univariate_ts_mode21_7K_0'))

    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t].reshape(10, 6) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([10, 1]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    final_preds = np.concatenate(final_preds, axis=1)


actual = test_y.reshape(10, 21)
predicted = final_preds
plt.plot(scaler_for_output.inverse_transform(actual.T), label="actual")
plt.plot(scaler_for_output.inverse_transform(predicted.T), label="predicted")
plt.legend()
plt.show()

total_MAE = 0.0
for i in range(test_len):
    total_MAE += mean_absolute_error(actual[i, :], predicted[i, :])
total_MAE /= test_len
print('The Mean Absolute Error is: {}'.format(total_MAE))


