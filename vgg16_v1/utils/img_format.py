""" Including all required execution for
    resizing image, image augmentation, image to array and so on.
"""
import tensorflow as tf
import numpy as np
import utils.shared as shared
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import glob
import os.path


class img_format:
    def __init__(self):
        self.size = (shared.IMAGE_SIZE, shared.IMAGE_SIZE)

    # https://keras.io/preprocessing/image/
    # shear/zoom etc. for augmentation
    def img_generator(self, _mode=None):
        if(_mode == 'train'):
            datagen = image.ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=90,
                            fill_mode='nearest',
                            shear_range=0.2,
                            zoom_range=0.2,
                            width_range=0.2,
                            height_range=0.2)
        else:
            datagen = image.ImageDataGenerator(rescale=1. / 255)
        return datagen

    def img_aug_folderflow(self, _generator, _batch, _path):
        img_generator = _generator.flow_from_directory(
                            _path,
                            target_size=self.size,
                            batch_size=_batch)
        return img_generator.class_indices, img_generator

    # Data augmentation of all models within brand's path
    def img_aug_device(self, _generator, _batch, _path):
        for device_name in _path:
            img_list = glob.glob(device_name+'/*')
            for img_name in img_list:
                self.img_adjust(img_name)
                img = load_img(img_name)
                img = img_to_array(img)
                img = img.reshape((1,)+img.shape)
                gen_data = _generator.flow(x=img,
                                           batch_sizes=1,
                                           shuffle=True,
                                           save_to_dir=device_name,
                                           save_format='jpeg')
                self.img_gen_save(gen_data, _batch)
                print('Augment '+os.path.basename(img_name))

    def img_gen_save(self, _gen_data, _batch):
        count = 0
        for batch in _gen_data:
            count += 1
            if(count > _batch):
                break

    def img_aug_flow(self, _generator, _batch, _path):
        dir_list = glob.glob(_path+'/*')
        for dir_name in dir_list:
            self.img_aug_device(_generator, _batch, dir_name)

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
        img_bg.save(_img)
        # return img_bg

    def img_array(self, _img):
        array = image.img_to_array(_img)
        array = np.expand_dims(array, axis=0)
        # preprocess_input should be used
        array = preprocess_input(array)
        return array
