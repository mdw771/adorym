import numpy as np
import tensorflow as tf


def cartesian_to_spherical(arr, dist_to_source_nm, psize_nm):
    """
    Convert 3D cartesian array into spherical coordinates.
    """
    # Cartesian array indices
    x_ind, y_ind, z_ind = [np.arange(arr.shape[0], dtype=int),
                           np.arange(arr.shape[1], dtype=int),
                           np.arange(arr.shape[2], dtype=int)]
    # Cartesian coordiantes with real unit
    x_true, y_true, z_true = [(x_ind - np.median(x_ind)) * psize_nm,
                              (y_ind - np.median(y_ind)) * psize_nm,
                              z_ind * psize_nm]
    cart_interp = RegularGridInterpolator((x_true, y_true, z_true), arr, bounds_error=False, fill_value=0)
    # Spherical array indices
    r_ind, theta_ind, phi_ind = [np.arange(arr.shape[0], dtype=int),
                                 np.arange(arr.shape[1], dtype=int),
                                 np.arange(arr.shape[2], dtype=int)]
    r_true = r_ind * psize_nm + dist_to_source_nm
    theta_true = (theta_ind - np.median(theta_ind)) * (2 * theta_max / (theta_ind.size - 1))
    phi_true = (phi_ind - np.median(phi_ind)) * (2 * phi_max / (phi_ind.size - 1))
    r, theta, phi = np.meshgrid(r_true, theta_true, phi_true)
    x_interp = r * np.cos(theta) * np.sin(phi)
    y_interp = r * np.sin(theta)
    z_interp = r * np.cos(theta) * np.cos(phi)
    z_interp -= dist_to_source_nm
    x_interp /= psize_nm
    y_interp /= psize_nm
    z_interp /= psize_nm
    coords_interp = np.vstack([x_interp.flatten(), y_interp.flatten(), z_interp.flatten()]).transpose()
    dat_interp = cart_interp(coords_interp)
    r_ind_mesh, theta_ind_mesh, phi_ind_mesh = np.meshgrid(r_ind, theta_ind, phi_ind)
    arr_sph = np.zeros_like(arr)
    arr_sph[r_ind_mesh.flatten(), theta_ind_mesh.flatten(), phi_ind_mesh.flatten()] = dat_interp

    return arr_sph, (r_true, theta_true, phi_true)


def biliniear_interpolation_3d(data, warp):
    """
    Interpolate a 3D array (monochannel).
    """
    n_pts = warp.shape[0]
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
