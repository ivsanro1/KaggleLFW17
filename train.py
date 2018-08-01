# from sklearn.model_selection import KFold
import numpy as np
import pdb
import argparse
from scipy.ndimage import rotate
import random
import time, sys

def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()





parser = argparse.ArgumentParser()
parser.add_argument("X",
                    nargs=1,
                    metavar="X_path",
                    help="Training data (features). Expected format: .npy")
parser.add_argument("y",
                    nargs=1,
                    metavar="y_path",
                    help="Training data (labels). Expected format: .npy")

parser.add_argument("--validate",
                    action="store_true",
                    help="Also validate the model accuracy when the training finishes.")
parser.add_argument("--validation_split_size",
                    help="percentage of training data that will be used for validation (it will not be used in training)",
                    nargs='?',
                    type=float)




# Called after parse args so if a arg parse error occurs,
# tf warnings and sklearn deprecation messages do not appear -> clear info
import tensorflow as tf 
from sklearn.cross_validation import StratifiedShuffleSplit



def get_random_rotation(angle):
    return random.uniform(-angle, angle)

def apply_random_rotation_fn(image, angle):
    rot = get_random_rotation(angle)
    return rotate(image, rot, reshape=False)

def apply_horizontal_flip_fn(image):
    return np.fliplr(image)

def data_augmentation(features,
                      labels,
                      n_augmentations_per_image=5,
                      apply_horizontal_flip=True,
                      horizontal_flip_chance=1.,
                      apply_random_rotations=True,
                      rotation_chance=1.,
                      max_rotation_angle=30):
    aux_features = features[:]
    aux_labels = labels[:]

    assert (features.shape[0] == labels.shape[0])

    for i in range(labels.shape[0]):
        for _ in range(n_augmentations_per_image):
            augmented_image = features[i].reshape((50, 37))
            

            if (random.uniform(0, 1) < rotation_chance):
                # print('deb')
                augmented_image = apply_random_rotation_fn(augmented_image, max_rotation_angle)


            if (random.uniform(0, 1) < horizontal_flip_chance):
                # pdb.set_trace()
                # print('deb2')
                augmented_image = apply_horizontal_flip_fn(augmented_image)
            
            aux_features = np.vstack([aux_features, np.asarray([augmented_image.flatten()])])
            aux_labels = np.append(aux_labels, labels[i]) # same label obv

            update_progress("Performing data augmentation", i/labels.shape[0])
    update_progress("Performing data augmentation", 1)
    
    return aux_features, aux_labels
    


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 50, 37, 1])

  bn_conv_0 = tf.layers.batch_normalization(inputs=input_layer)
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=bn_conv_0,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.leaky_relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  bn_conv_1 = tf.layers.batch_normalization(inputs=pool1)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=bn_conv_1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.leaky_relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  bn_conv_2 = tf.layers.batch_normalization(inputs=pool2)
  # Convolutional Layer #3 and Pooling Layer #3
  conv3 = tf.layers.conv2d(
      inputs=bn_conv_2,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.leaky_relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten layer
  pool3_flat = tf.reshape(pool3, [-1, 6 * 4 * 256])

  bn1 = tf.layers.batch_normalization(inputs=pool3_flat)
  # Dense Layer #1
  dense1 = tf.layers.dense(inputs=bn1, units=1024, activation=tf.nn.leaky_relu)

  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.725, training=mode == tf.estimator.ModeKeys.TRAIN)

  
  bn2 = tf.layers.batch_normalization(inputs=dropout1)
  # Dense Layer #2
  dense2 = tf.layers.dense(inputs=bn2, units=1024, activation=tf.nn.leaky_relu)
  

  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.725, training=mode == tf.estimator.ModeKeys.TRAIN)

  bn3 = tf.layers.batch_normalization(inputs=dropout2)
  # Dense Layer #3
  dense3 = tf.layers.dense(inputs=bn3, units=1024, activation=tf.nn.relu)
  dropout3 = tf.layers.dropout(
      inputs=dense3, rate=0.725, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout3, units=7)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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
    args_ = parser.parse_args()

    if (args_.validate and args_.validation_split_size == None):
        print("\n\nPlease specify a validation split size if you want to validate the model.")
        sys.exit(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Load tr data as np
    X = np.load(args_.X[0])
    y = np.load(args_.y[0])

    X_train = X[:]
    y_train = y[:]

    if (args_.validate):
        stratSplit = StratifiedShuffleSplit(y, 1, test_size=args_.validation_split_size, random_state=666)
        StratifiedShuffleSplit(y, n_iter=1, test_size=args_.validation_split_size)
        for train_idx, test_idx in stratSplit:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test  = X[test_idx]
            y_test  = y[test_idx]
        
    X_train, y_train = data_augmentation(X_train,
                                         y_train,
                                         n_augmentations_per_image=30,
                                         max_rotation_angle=20,
                                         horizontal_flip_chance=0.5,
                                         rotation_chance=0.95)

    print("training with %d images" % (X_train.shape[0]))
    # pdb.set_trace()
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
        steps=30000)
    #     hooks=[logging_hook] log training procedure

    if (args_.validate):
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

if __name__ == "__main__":
    main(0)