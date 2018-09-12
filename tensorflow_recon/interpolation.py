import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

from constants import *


def cartesian_to_spherical(arr, dist_to_source_nm, psize_nm, theta_max=PI/18, phi_max=PI/18, interpolation='nearest'):
    """
    Convert 3D cartesian array into spherical coordinates.
    """
    # Spherical array indices
    arr = tf.cast(arr, dtype=tf.float32)
    arr_shape_float = tf.cast(arr.get_shape(), dtype=tf.float32)
    theta_ind, phi_ind, r_ind = [tf.cast(tf.range(arr.get_shape()[0]), dtype=tf.float32),
                                 tf.cast(tf.range(arr.get_shape()[1]), dtype=tf.float32),
                                         tf.cast(tf.range(arr.get_shape()[2]), dtype=tf.float32)]
    theta_mid = (arr_shape_float[0] - 1) / 2
    phi_mid = (arr_shape_float[1] - 1) / 2
    r_true = r_ind + dist_to_source_nm / psize_nm
    theta_true = (theta_ind - theta_mid) * (2 * theta_max / (arr_shape_float[0] - 1))
    phi_true = (phi_ind - phi_mid) * (2 * phi_max / (arr_shape_float[1] - 1))
    phi, theta, r = tf.meshgrid(phi_true, theta_true, r_true)
    x_interp = r * tf.sin(theta) + theta_mid
    y_interp = r * tf.cos(theta) * tf.sin(phi) + phi_mid
    z_interp = r * tf.cos(theta) * tf.cos(phi)
    z_interp -= dist_to_source_nm / psize_nm
    coords_interp = tf.transpose(tf.stack([tf.reshape(x_interp, [-1]), tf.reshape(y_interp, [-1]), tf.reshape(z_interp, [-1])]))
    if interpolation == 'trilinear':
        coords_interp = tf.clip_by_value(coords_interp, 0, tf.reduce_min(arr_shape_float) - 2)
        dat_interp = triliniear_interpolation_3d(arr, coords_interp)
    elif interpolation == 'nearest':
        coords_interp = tf.round(coords_interp)
        coords_interp = tf.clip_by_value(coords_interp, 0, tf.reduce_min(arr_shape_float) - 1)
        coords_interp = tf.cast(coords_interp, tf.int32)
        dat_interp = tf.gather_nd(arr, coords_interp)
    else:
        raise ValueError('Interpolation must be \'trilinear\' or \'nearest\'.')
    arr_sph = tf.reshape(dat_interp, arr.get_shape())

    return arr_sph, (r_true, theta_true, phi_true)


def triliniear_interpolation_3d(data, warp):
    """
    Interpolate a 3D array (monochannel).
    """
    n_pts = warp.shape[0]
    arr_shape = data.get_shape().as_list()
    warp = tf.cast(warp, tf.float32)
    i000 = tf.cast(tf.floor(warp), dtype=tf.int32)
    i100 = i000 + tf.constant([1, 0, 0])
    i010 = i000 + tf.constant([0, 1, 0])
    i001 = i000 + tf.constant([0, 0, 1])
    i110 = i000 + tf.constant([1, 1, 0])
    i101 = i000 + tf.constant([1, 0, 1])
    i011 = i000 + tf.constant([0, 1, 1])
    i111 = i000 + tf.constant([1, 1, 1])
    c000 = tf.gather_nd(data, i000)
    c100 = tf.gather_nd(data, i100)
    c010 = tf.gather_nd(data, i010)
    c001 = tf.gather_nd(data, i001)
    c110 = tf.gather_nd(data, i110)
    c101 = tf.gather_nd(data, i101)
    c011 = tf.gather_nd(data, i011)
    c111 = tf.gather_nd(data, i111)
    # build matrix
    h00 = tf.ones(n_pts)
    x0 = tf.cast(i000[:, 0], dtype=tf.float32)
    y0 = tf.cast(i000[:, 1], dtype=tf.float32)
    z0 = tf.cast(i000[:, 2], dtype=tf.float32)
    x1 = tf.cast(i111[:, 0], dtype=tf.float32)
    y1 = tf.cast(i111[:, 1], dtype=tf.float32)
    z1 = tf.cast(i111[:, 2], dtype=tf.float32)
    h1 = tf.stack([h00, x0, y0, z0, x0 * y0, x0 * z0, y0 * z0, x0 * y0 * z0])
    h2 = tf.stack([h00, x1, y0, z0, x1 * y0, x1 * z0, y0 * z0, x1 * y0 * z0])
    h3 = tf.stack([h00, x0, y1, z0, x0 * y1, x0 * z0, y1 * z0, x0 * y1 * z0])
    h4 = tf.stack([h00, x1, y1, z0, x1 * y1, x1 * z0, y1 * z0, x1 * y1 * z0])
    h5 = tf.stack([h00, x0, y0, z1, x0 * y0, x0 * z1, y0 * z1, x0 * y0 * z1])
    h6 = tf.stack([h00, x1, y0, z1, x1 * y0, x1 * z1, y0 * z1, x1 * y0 * z1])
    h7 = tf.stack([h00, x0, y1, z1, x0 * y1, x0 * z1, y1 * z1, x0 * y1 * z1])
    h8 = tf.stack([h00, x1, y1, z1, x1 * y1, x1 * z1, y1 * z1, x1 * y1 * z1])
    h = tf.stack([h1, h2, h3, h4, h5, h6, h7, h8])
    h = tf.transpose(h, perm=[2, 0, 1])
    c = tf.transpose(tf.stack([c000, c100, c010, c110, c001, c101, c011, c111]))
    c = tf.expand_dims(c, -1)
    a = tf.squeeze(tf.matmul(tf.matrix_inverse(h), c))

    x = warp[:, 0]
    y = warp[:, 1]
    z = warp[:, 2]
    f = a[:, 0] + a[:, 1] * x + a[:, 2] * y + a[:, 3] * z + \
        a[:, 4] * x * y + a[:, 5] * x * z + a[:, 6] * y * z + a[:, 7] * x * y * z
    return f
