# -*- coding: utf-8 -*-

"""Main module."""
import utils.base as bs
import utils.shared as shared
import tensorflow as tf


base = bs.Base()
train_url = shared.TRAIN_URL
test_url = shared.TEST_URL
is_url = True
label = shared.LABEL

# Call load_data() to parse the CSV file.
(train_feature, train_label, test_feature, test_label) = base.load_data_csv(
    label, is_url, train_url, test_url)

# Create feature columns for all features.
feature_cols = base.get_feature_cols(train_feature)

# Select premade model from estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_cols,
    hidden_units=[10, 10],
    n_classes=3)
# Train model
classifier.train(
    input_fn=lambda: base.train_input_fn(train_feature, train_label, 10),
    steps=1000)
# Evaluate model
eval_result = classifier.evaluate(
    input_fn=lambda: base.eval_input_fn(test_feature, test_label, 10))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
