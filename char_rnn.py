import tensorflow as tf
from params import FLAGS

class CharRnn(object):
    """
    cnn for char-level word embeding
    """

    def __init__(self, chars, chars_size):
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length")

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(FLAGS.lstm_hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(FLAGS.lstm_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, # inputs
                sequence_length=sequence_length, dtype=tf.float32)
            self.output = tf.concat([output_fw, output_bw], axis=-1)
            self.output = tf.nn.dropout(output, dropout_keep_prob)