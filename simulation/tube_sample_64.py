from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import dxchange
import time
import tifffile

from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator


def test_model_prop_pipeline():

    n_particles = 12
    top_y = 6.e-7
    top_radius = 2.e-7
    bottom_radius = 24.e-7
    top_thickness = 2.e-7
    bottom_thickness = 4.e-7
    length = 50.e-7
    bottom_y = top_y + length

    silicon = XraylibMaterial('Si', 2.33)
    titania = XraylibMaterial('TiO2', 4.23)
    air = CustomMaterial(delta=0, beta=0)

    try:
        # raise ValueError
        grid_delta = np.load('data/sav/grid/grid_delta.npy')
        grid_beta = np.load('data/sav/grid/grid_beta.npy')
    except:

        tube0 = TruncatedCone_3d(top_center=Point([32e-7, top_y, 32e-7]),
                                 length=length,
                                 top_radius=top_radius,
                                 bottom_radius=bottom_radius)
        phantom = Phantom(geometry=tube0, material=silicon)

        tube1 = TruncatedCone_3d(top_center=Point([32e-7, top_y, 32e-7]),
                                 length=length,
                                 top_radius=top_radius-top_thickness,
                                 bottom_radius=bottom_radius-bottom_thickness)
        tube1 = Phantom(geometry=tube1, material=air)
        phantom.children.append(tube1)

        rand_y = []
        for i in range(n_particles):
            xi = np.random.rand()
            rand_y.append((top_radius - np.sqrt(top_radius ** 2 - top_radius ** 2 * xi + bottom_radius ** 2 * xi)) /
                          (top_radius - bottom_radius) * length + top_y)

        for part_y in rand_y:
            r = top_radius + (bottom_radius - top_radius) / (length - 1) * (part_y - top_y)
            theta = np.random.rand() * np.pi * 2
            part_x = np.cos(theta) * r + 32e-7
            part_z = np.sin(theta) * r + 32e-7
            rad = int(np.random.rand() * 6e-7) + 4e-7
            sphere = Sphere_3d(center=Point([part_x, part_y, part_z]),
                               radius=rad)
            sphere = Phantom(geometry=sphere, material=titania)
            phantom.children.append(sphere)

        grid_data = discrete_phantom(phantom, 1e-7, [[0, 63e-7]] * 3, prop=['delta', 'beta'], ratio=1,
                                     mkwargs={'energy': 5}, overlay_mode='replace')
        grid_delta = grid_data[..., 0]
        grid_beta = grid_data[..., 1]
        dxchange.write_tiff(grid_delta*1e7, 'tmp', dtype='float32', overwrite=True)

        np.save('data/sav/grid/64/grid_delta.npy', grid_delta)
        np.save('data/sav/grid/64/grid_beta.npy', grid_beta)

    sim = Simulator(energy=5000,
                    grid=(grid_delta, grid_beta),
                    psize=[1e-7, 1e-7, 1e-7])

    # print(sim.lmbda_nm)
    sim.initialize_wavefront('plane')
    # sim.initialize_wavefront('point_projection_lens', focal_length=2e6, lens_sample_dist=4e6)
    # sim.initialize_wavefront('spherical', dist_to_source=9e12)
    # plt.imshow(np.angle(sim.wavefrsont))
    # plt.show()
    t0 = time.time()
    wavefront = sim.multislice_propagate(free_prop_dist=1e-4)
    print('Propagation time: {} ms'.format((time.time() - t0) * 1000))

    # dxchange.write_tiff(sim.wavefront, 'exiting_sphwave_flattened', dtype='float32', overwrite=True)
    plt.imshow(np.abs(wavefront))
    plt.show()


if __name__ == '__main__':
    test_model_prop_pipeline()