from util import *
import tensorflow as tf
import numpy as np

obj = np.array([[[[1, 1.5], [2, 2.5]], [[3, 3.5], [4, 4.5]]], [[[5, 5.5], [6, 6.5]], [[7, 7.5], [8, 8.5]]]])
obj = tf.convert_to_tensor(obj)
# coord_ls = save_rotation_lookup([64, 64, 64], 500)
coord_old = read_origin_coords('arrsize_2_2_2_ntheta_10', 1)
res = apply_rotation(obj, coord_old, 'arrsize_2_2_2_ntheta_10')
sess = tf.Session()
print(sess.run(res))
# print(coord_ls[250])
