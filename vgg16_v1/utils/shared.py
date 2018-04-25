""" Macro definitions """


SIZE = 224
""" better to be updated by reading config files """
LABEL = 'DeviceModels'
CLASS_NUM = 14
DeviceModels = range(CLASS_NUM)

# CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
#                     'PetalLength', 'PetalWidth', 'Species']
# SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
# TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
# LABEL = 'Species'
#
# FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]

RESULT_PATH = './results'
MODEL_FILE = './results/vgg_model.json'
WEIGHT_FILE = './results/vgg_weight.h5'

LEARNING_RATE = 0.01
BATCH_SIZE = 10
