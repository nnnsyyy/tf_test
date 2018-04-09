# -*- coding: utf-8 -*-

"""Main module."""
import utils.base as bs
import utils.shared as shared
import model.model as md


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
classifier = md.Model(feature_cols)
# Train model
classifier.model_train(train_feature, train_label)
# Evaluate model
classifier.model_eval(test_feature, test_label)
