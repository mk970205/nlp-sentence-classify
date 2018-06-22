import tensorflow as tf
from util import load_data
from tensorflow.contrib import learn
tf.flags.DEFINE_string("pos_data", "./data/MR/rt-polarity.pos")
tf.flags.DEFINE_string("neg_data", "./data/MR/rt-polarity.neg")

FLAGS = tf.flags.FLAGS

def preprocess():
    print("loading data...")
    text, label = load_data(FLAGS.pos_data_dir, FLAGS.neg_data_dir)

    #vocab
    max_doc_length = max([len(x.split(" ")) for x in text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)
