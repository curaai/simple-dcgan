import numpy as np
import os
import pickle

import scipy.misc
from sklearn.utils import shuffle


def dump_image(path, output_path):
    cats = os.listdir(path)
    cats = list(map(lambda x: os.path.join(path, x), cats))[:1000]

    images = list(map(lambda x: scipy.misc.imread(x), cats))
    images = np.array(images)
    images = shuffle(images, random_state=4)
    with open(output_path, 'wb') as f:
        pickle.dump(images, f)

    return images


def get_from_dump(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return shuffle(data, random_state=10)

