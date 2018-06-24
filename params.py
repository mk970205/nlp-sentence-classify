import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding (default: 128)")
tf.flags.DEFINE_string("cnn_filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("cnn_num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("lstm_hidden_size", 100, "Size of lstm hidden layer.")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

#preprocessing parameters
tf.flags.DEFINE_float("CV_percentage", .1, "Percentage of the testing data to use for cross validation")
tf.flags.DEFINE_string("pos_data_dir", "./data/MR/rt-polarity.pos", "dir of positive data")
tf.flags.DEFINE_string("neg_data_dir", "./data/MR/rt-polarity.neg", "dir of negative data")

#training Session parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#eval parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")