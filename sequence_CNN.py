import tensorflow as tf
import numpy as np

class sequenceCNN:
    def __init__(self, vocab_processor, vocab_size, embedding_size, sequence_length, word_length, vocab_size_char,
                 embedding_size_char, filter_sizes, num_filters, num_classes=2, num_char_filters=1):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_x_sub = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length, word_length], name="input_x_sub")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

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
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for _, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")