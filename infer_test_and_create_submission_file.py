import numpy as np
import pdb
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("X",
                    nargs=1,
                    metavar="X_path",
                    help="Test data to infer (features). Expected format: .npy")

args_ = parser.parse_args()

from train import cnn_model_fn
import tensorflow as tf

X_test = np.load(args_.X[0])

# Estimator
face_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="faces_convnet_model")

infer_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    shuffle=False)
infer_results = face_classifier.predict(input_fn=infer_input_fn)

predicted_classes = [el['classes'] for el in infer_results]

# zip with the column of test element number
submission = [list(el) for el in zip([i for i in range(X_test.shape[0])], predicted_classes)]

# add at the head the header
submission.insert(0, ['ImageId', 'PredictedClass'])

# print(submission)


with open("submission.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(submission)

print("\n\nSubmission file 'submission.csv' created.")