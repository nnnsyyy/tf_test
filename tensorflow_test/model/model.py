import tensorflow as tf
import utils.shared as shared


class Model:
    def __init__(self, _feature_cols):
        self.feature_cols = _feature_cols
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=self.feature_cols,
            hidden_units=[10, 10],
            n_classes=3)

    def model_train(self, _train_feature, _train_label):
        # Train model
        self.classifier.train(
            input_fn=lambda: self.train_input_fn(
                _train_feature, _train_label, 10),
            steps=1000)

    def train_input_fn(self, _features, _labels, _batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(_features), _labels))
        # print(dataset)
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(_batch_size)
        # print(dataset)
        # Return the dataset.
        return dataset

    def model_eval(self, _test_feature, _test_label):
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.eval_input_fn(
                _test_feature, _test_label, 10))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    def eval_input_fn(self, _features, _labels, _batch_size):
        """An input function for evaluation or prediction"""
        features = dict(_features)
        if _labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, _labels)
            # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        # Batch the examples
        assert _batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(_batch_size)
        return dataset

    def model_predict(self, _predict_x, _expected):
        predictions = self.classifier.predict(
            input_fn=lambda: self.eval_input_fn(
                _predict_x, _labels=None, _batch_size=10))
        for pred_dict, expec in zip(predictions, _expected):
            template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print(template.format(
                shared.SPECIES[class_id], 100 * probability, expec))
