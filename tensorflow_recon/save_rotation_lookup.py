from util import *
import tensorflow as tf
import numpy as np

# obj = np.array([[[[1, 1], [2, 2]], [[3, 3], [4, 4]]], [[[5, 5], [6, 6]], [[7, 7], [8, 8]]]])
# obj = tf.convert_to_tensor(obj)
coord_ls = save_rotation_lookup([64, 64, 64], 500)
# coord_old = read_origin_coords('arrsize_2_2_2_ntheta_10', 1)
# apply_rotation(obj, coord_old, 'arrsize_2_2_2_ntheta_10')
# print(coord_ls[250])
