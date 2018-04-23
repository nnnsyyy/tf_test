""" Basic functions for tf pipeline """

# import tensorflow as tf
# import pandas as pd
# import numpy as np
from . import shared
from . import img_format
# import vggmodel.vggmodel as VGGModel
# import numpy as np


class Base:
    def __init__(self):
        self.size = (shared.IMAGE_SIZE, shared.IMAGE_SIZE)

    # https://keras.io/preprocessing/image/
    # shear/zoom etc. for augmentation
    def img_import(self, _img):
        img = img_format.img_adjust(_img)
        return img_format.img_array(img)
