import tensorflow as tf
from util import load_data
from util import batch_iter
from tensorflow.contrib import learn
import numpy as np
import time
import os
import datetime
from params import FLAGS
from sequence_RNN import sequenceRNN
from sequence_CNN import sequenceCNN
def preprocess():
    print("loading data...")
    text, label = load_data(FLAGS.pos_data_dir, FLAGS.neg_data_dir)

    #vocab
    max_doc_length = max([len(x.split(" ")) for x in text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)
    x = np.array(list(vocab_processor.fit_transform(text)))

    np.random.seed(10)
    shuffle_index = np.random.permutation(np.arange(len(label)))
    x_shuffled = x[shuffle_index]
    y_shuffled = label[shuffle_index]

    CV_index = -1 * int(FLAGS.CV_percentage * float(len(label)))
    x_train, x_test = x_shuffled[:CV_index], x_shuffled[CV_index:]
    y_train, y_test = y_shuffled[:CV_index], y_shuffled[CV_index:]

    del x, label, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return x_train, y_train, vocab_processor, x_test, y_test

def train(x_train, y_train, vocab_processor, x_test, y_test, model="CNN"):
    config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if model == "RNN":
            model = sequenceRNN(
                sequence_length=x_train.shape[1],
                vocab_processor=vocab_processor,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.lstm_hidden_size
            )

        elif model == "CNN":
            model = sequenceCNN(
                sequence_length=x_train.shape[1],
                vocab_processor=vocab_processor,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.cnn_filter_sizes.split(","))),
                num_filters=FLAGS.num_filters
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            x_list = np.array(list(x_batch))

            x_word_ids = []
            for _1 in range(x_list.shape[0]):
                ids = []
                for _2 in range(x_list.shape[1]):
                    word_tuple = list(sorted_vocab[x_list[_1][_2]])
                    ids.append(word_tuple[0])
                x_word_ids.append(ids)

            feed_dict = {
                model.input_x: x_batch,
                model.input_x_sub: np.array(x_word_ids),
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob 
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """

            x_list = np.array(list(x_batch))

            x_word_ids = []
            for _1 in range(x_list.shape[0]):
                ids = []
                for _2 in range(x_list.shape[1]):
                    word_tuple = list(sorted_vocab[x_list[_1][_2]])
                    ids.append(word_tuple[0])
                x_word_ids.append(ids)

            feed_dict = {
                model.input_x: x_batch,
                model.input_x_sub: np.array(x_word_ids),
                model.input_y: y_batch,
                model.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run([global_step, test_summary_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                test_step(x_test, y_test, writer=test_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_test, y_test = preprocess()
    train(x_train, y_train, vocab_processor, x_test, y_test)

#if __name__ == '__main__':
   # tf.app.run()

x_train, y_train, vocab_processor, x_test, y_test = preprocess()
model = sequenceRNN(
                sequence_length=x_train.shape[1],
                vocab_processor=vocab_processor,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.lstm_hidden_size
            )
batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
for batch in batches:
    x_batch, y_batch = zip(*batch)
    x_list = np.array(list(x_batch))
    print(x_list.shape)
    x_word_ids = []
    for _1 in range(x_list.shape[0]):
        ids = []
        for _2 in range(x_list.shape[1]):
            word_tuple = list(sorted_vocab[x_list[_1][_2]])
            ids.append(word_tuple[0])
        x_word_ids.append(ids)
    print(x_word_ids)
    break
