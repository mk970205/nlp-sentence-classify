#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from util import load_data, batch_iter
from tensorflow.contrib import learn
import csv
from itertools import chain, repeat, islice
# Parameters
# ==================================================

positive_data_file = "./data/MR/rt-polarity.pos"
negative_data_file = "./data/MR/rt-polarity.neg"

batch_size = 64
checkpoint_dir = "./runs/1529914597/checkpoints/"
eval_train = True

allow_soft_placement = True
log_device_placement = False
max_word_length = 20
# CHANGE THIS: Load data. Load your own data here

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)
   
def char_to_id(charlist, char_dict):
    idlist = []
    for i in range(len(charlist)):
        char = charlist[i].lower()
        id = char_dict.index(char)
        idlist.append(id)

    return idlist



if eval_train:
    x_raw, y_test = load_data(positive_data_file, negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

char_dict = []
for sent in x_raw:
    for c in sent:
        if c not in char_dict:
            char_dict.append(c)
if '' not in char_dict:
    char_dict.append('')

# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_x_sub = graph.get_operation_by_name("input_x_sub").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            x_list = np.array(list(x_test_batch))

            x_word_ids = []
            for _1 in range(x_list.shape[0]):
                ids = []
                for _2 in range(x_list.shape[1]):
                    word_tuple = list(sorted_vocab[x_list[_1][_2]])
                    if word_tuple[0] != "<UNK>":
                        char_list = list(word_tuple[0])
                    else:
                        char_list = []
                    id_list = char_to_id(char_list, char_dict)
                    ids.append(list(pad(id_list, max_word_length, char_dict.index(''))))
                x_word_ids.append(ids)

            batch_predictions = sess.run(predictions, {input_x: x_test_batch, input_x_sub:x_word_ids, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)