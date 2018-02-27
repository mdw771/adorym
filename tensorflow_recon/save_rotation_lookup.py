from util import *
import tensorflow as tf


coord_ls = save_rotation_lookup([64, 64, 64], 500)
sess = tf.Session()
print(sess.run(coord_ls[0]))
