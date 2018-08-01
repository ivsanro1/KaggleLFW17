# from sklearn.model_selection import KFold
import numpy as np
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-X",
                    nargs=None,
                    metavar="path_tr_features",
                    help="Training data (features). Expected format: .npy")
parser.add_argument("-y",
                    nargs=None,
                    metavar="path_tr_labels",
                    help="Training data (labels). Expected format: .npy")

parser.add_argument("--validate",
                    action="store_true",
                    help="Also validate the model accuracy when the training finishes.")
parser.add_argument("validation_split_size",
                    help="percentage of training data that will be used for validation (it will not be used in training)",
                    const=0.2,
                    nargs='?',
                    type=float)

args_ = parser.parse_args()


# Called after parse args so if a arg parse error occurs,
# tf warnings and sklearn deprecation messages do not appear -> clear info
import tensorflow as tf 
from sklearn.cross_validation import StratifiedShuffleSplit


tf.logging.set_verbosity(tf.logging.INFO)





def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 50, 37, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3 and Pooling Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Dense Layer #1
  pool3_flat = tf.reshape(pool3, [-1, 6 * 4 * 256])
  dense1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.95, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2
  dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.95, training=mode == tf.estimator.ModeKeys.TRAIN)

#   # Dense Layer #3
#   dense3 = tf.layers.dense(inputs=dropout2, units=1024, activation=tf.nn.relu)
#   dropout3 = tf.layers.dropout(
#   inputs=dense3, rate=0.75, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout2, units=7)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



## START
def main(unused_argv):
    # Load tr data as np
    X = np.load("data/X_train.npy")
    y = np.load("data/y_train.npy")

    stratSplit = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=666)
    StratifiedShuffleSplit(y, n_iter=1, test_size=0.2)
    for train_idx, test_idx in stratSplit:
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test  = X[test_idx]
        y_test  = y[test_idx]
        

    # Validate that data is consistent. Each row: one image with a class
    assert (X_train.shape[0] == y_train.shape[0])

    X_train = X_train.reshape(X_train.shape[0], 50, 37, 1)
    y_train = np.asarray(y_train, dtype=np.int32)

    # pdb.set_trace()
    
    # Estimator
    face_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="faces_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    face_classifier.train(
        input_fn=train_input_fn,
        steps=10000)
    #     hooks=[logging_hook] log training procedure

    # Evaluate the model and print results (TRAINING, to see if it overfits)
    eval_input_fn_tr = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        num_epochs=1,
        shuffle=False)
    eval_results_tr = face_classifier.evaluate(input_fn=eval_input_fn_tr)
    print("\n\n\n\n\nTRAINING ACC:\n\n\n")
    print(eval_results_tr)

    print('\n\n\n#################################################\n\n\n')

    # Evaluate the model and print results (VALIDATION)
    eval_input_fn_val = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results_val = face_classifier.evaluate(input_fn=eval_input_fn_val)
    print("\n\n\n\n\nVALIDATION ACC:\n\n\n")
    print(eval_results_val)

main(0)