""" Including all required execution for
    resizing image, image augmentation, image to array and so on.
"""
import tensorflow as tf
import numpy as np
import utils.shared as shared
from PIL import Image
from tf.python.keras.preprocessing import image
from tf.python.keras.applications.vgg16 import preprocess_input


class img_format:
    def __init__(self):
        self.size = (shared.IMAGE_SIZE, shared.IMAGE_SIZE)

    # https://keras.io/preprocessing/image/
    # shear/zoom etc. for augmentation
    def img_generator(self, _train_dir, _val_dir):
        train_datagen = image.ImageDataGenerator(
                            rescale=1./255,
                            fill_mode='nearest',
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)
        test_datagen = image.ImageDataGenerator(
                            rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                            _train_dir,
                            target_size=self.size,
                            batch_size=shared.BATCH_SIZE,
                            class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(
                                _val_dir,
                                target_size=self.size,
                                batch_size=shared.BATCH_SIZE,
                                class_mode='categorical')
        return train_generator, test_generator

    # resize images into (224, 224) with white bg
    # _img: path of image
    def img_adjust(self, _img):
        # scaling img into (224, 224)
        img = Image.open(_img)
        img.thumbnail(self.size, Image.ANTIALIAS)
        # add white bg
        img_bg = Image.new('RGB', self.size, (255, 255, 255))
        img_bg.paste(
            img, (int((self.size[0] - img.size[0]) / 2),
                  int((self.size[1] - img.size[1]) / 2))
        )
        return img_bg

    def img_array(self, _img):
        array = image.img_to_array(_img)
        array = np.expand_dims(array, axis=0)
        # preprocess_input should be used
        array = preprocess_input(array)
        return array
