import tensorflow as tf
import numpy as np
class sequenceRNN:
    def __init__(self, vocab_processor, sequence_length, word_length, vocab_size, vocab_size_char,
                 embedding_size, embedding_size_char, hidden_size=100, num_classes=2, num_char_filters=1):
        
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_x_sub = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length, word_length], name="input_x_sub")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

        text_length = self._length(self.input_x)

        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Word_embedding")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("char_embedding"):
            self.W_char = tf.Variable(tf.random_uniform([vocab_size_char, embedding_size_char], -1.0, 1.0), name="Char_embedding")
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W_char, self.input_x_sub)
            s_charemb = tf.shape(self.embedded_chars2)
            char_embeddings = tf.reshape(self.embedded_chars2, shape=[s_charemb[0] * s_charemb[1], word_length, embedding_size_char, num_char_filters])
            Filter1 = tf.Variable(tf.truncated_normal(shape=[4, embedding_size_char,1, num_char_filters],stddev=0.1)) #(Filter_width, Embedding size, Layer Size, Number of filters)
            Bias1 = tf.Variable(tf.truncated_normal(shape=[num_char_filters],stddev=0.1)) #(Number of filters)
            Conv1 = tf.nn.conv2d(char_embeddings, Filter1, strides=[1, 1, 1, 1], padding='SAME') + Bias1 
            Activation1 = tf.nn.relu(Conv1)
            Pool1 = tf.nn.max_pool(Activation1, ksize=[1, word_length, 1, 1], strides=[1, word_length, 1, 1], padding='SAME')
            Pool1 = tf.squeeze(Pool1)
            output = tf.reshape(Pool1, shape = [-1, sequence_length, embedding_size_char]) #(batch size, max length of sentence, embeddings size)
            self.final_embed = tf.concat([self.embedded_chars, output], 2)
            

        with tf.name_scope("LSTM"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, name="1st_Cell")
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.final_embed, sequence_length=text_length, dtype=tf.float32)
            self.h_outputs = self.last_relevant(outputs, text_length)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        
    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)
