""" Model class
    Customized VGG16 (pre-trained weight of conv layers)
    Alter the last fc layer (class_num)
    Thanks to:
    http://cv-tricks.com/keras/fine-tuning-tensorflow/
"""

import tensorflow as tf
import utils.shared as shared
import model.img_format.img_format as img_format
import tensorflow.python.keras as K
from K.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from K.preprocessing import image
from K.layers import Input, Flatten, Dense
from K.models import Model, model_from_json
import os


class VGGModel:
    def __init__(self):
        self.shape = (shared.SIZE, shared.SIZE, 3)
        self.has_model = True if os.path.isfile(shared.MODEL_FILE) else False
        self.model = None

    def model_init(self, _mode=None):
        if(_mode == 'load' and self.has_model):
            self.model_load()
        else:
            print('Init model.')
            self.model_get()

    def model_get(self):
        vgg16_conv = VGG16(weights='imagenet',
                           input_shape=self.shape,
                           include_top=False)
        # freeze the conv layers
        for layer in vgg16_conv.layers:
            layer.trainable = False
        vgginput = Input(shape=self.shape, name='image_input')
        output_vgg16_conv = vgg16_conv(vgginput)

        with tf.variable_scope("New_FCs"):
            # Add the fully-connected layers
            x = Flatten(name='flatten')(output_vgg16_conv)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(shared.CLASS_NUM,
                      activation='softmax',
                      name='predictions')(x)

        # Create your own model
        # Weights and layers from VGG part will be hidden
        updatable_variables = tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES,
                                scope='New_FCs')
        my_model = Model(inputs=vgginput, outputs=x)
        my_model = self.model_train(my_model, x, updatable_variables)
        my_model.summary()
        self.model_save(my_model)
        return my_model

    def model_save(self, _model):
        if not os.path.isdir(shared.RESULT_PATH):
            os.makedirs(shared.RESULT_PATH)
        model_json = _model.to_json()
        with open(shared.MODEL_FILE, "w") as json_file:
            json_file.write(model_json)
        _model.save_weights(shared.WEIGHT_FILE)
        print("Saved model to disk")

    def model_load(self):
        json_file = open(os.path.join(shared.MODEL_FILE), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(shared.WEIGHT_PATH))
        print("Loaded model from disk")
        loaded_model.summary()
        # return loaded_model
        self.model = loaded_model

    # def model_train(self, _train_feature, _train_label):
    def mode_train(self, _model, _x, updatable_variables):
        # Train model
        with tf.variable_scope("Optimise"):
            # ## optimization and cost ops
            label = tf.placeholder(tf.int32,
                                   shape=(None), name='label')
            one_hot_y = tf.one_hot(label,
                                   shared.CLASS_NUM, name='onehot_output')
            loss = tf.losses.softmax_cross_entropy(one_hot_y, _x)

            tf.summary.scalar('loss', loss)
            # optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate)
            optimiser = tf.train.MomentumOptimizer(shared.LEARNING_RATE, 0.9)
            training_operation = optimiser.minimize(
                                    loss,
                                    var_list=updatable_variables)

        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='Optimise')

    # def train_input_fn(self, _features, _labels, _batch_size):
    #     """An input function for training"""
    #     # Convert the inputs to a Dataset.
    #     dataset = tf.data.Dataset.from_tensor_slices(
    #         (dict(_features), _labels))
    #     # print(dataset)
    #     # Shuffle, repeat, and batch the examples.
    #     dataset = dataset.shuffle(1000).repeat().batch(_batch_size)
    #     # print(dataset)
    #     # Return the dataset.
    #     return dataset

    # def model_eval(self, _test_feature, _test_label):
    #     eval_result = self.classifier.evaluate(
    #         input_fn=lambda: self.eval_input_fn(
    #             _test_feature, _test_label, 10))
    #     print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # def eval_input_fn(self, _features, _labels, _batch_size):
    #     """An input function for evaluation or prediction"""
    #     features = dict(_features)
    #     if _labels is None:
    #         # No labels, use only features.
    #         inputs = features
    #     else:
    #         inputs = (features, _labels)
    #         # Convert the inputs to a Dataset.
    #     dataset = tf.data.Dataset.from_tensor_slices(inputs)
    #     # Batch the examples
    #     assert _batch_size is not None, "batch_size must not be None"
    #     dataset = dataset.batch(_batch_size)
    #     return dataset

    def model_predict(self, _predict_img, _expected=None):
        img_array = self.img_array(_predict_img)
        # TODO Maybe cannot applied to the same way
        probabilities = self.model.predict(img_array)
        predictions = decode_predictions(probabilities)

        print(predictions)
