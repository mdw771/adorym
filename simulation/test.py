import numpy as np
import matplotlib.pyplot as plt
import dxchange

from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator


grid_delta = np.load('data/sav/grid/grid_delta.npy')
grid_beta = np.load('data/sav/grid/grid_beta.npy')

sim = Simulator(energy=5000,
                grid=(grid_delta, grid_beta),
                psize=[1e-7, 1e-7, 1e-7])

# wavefront = np.load('exiting.npy')

# wavefront = dxchange.read_tiff('phantom.tiff')

# wavefront = np.zeros([512, 512])
# wavefront[236:276, 236:276] = 1
# wavefront = 1 - wavefront
# wavefront = wavefront - wavefront[0, 0]
"""
distance for 2 algos to have same sampling rate: 1.0322e-3
analytical: Dz = dx^2 * N / lambda
"""
# wavefront = far_propagate(sim, wavefront, 1e-3, pad=None)


# wavefront = np.fft.fftshift(np.fft.fft2(wavefront))


# wavefront = propagate_tf(sim, wavefront, -1e-3)
# sim.multislice_propagate()


# wavefront = wavefront[203:309, 203:309]
# plt.figure()
r = np.log(np.abs(wavefront))
plt.imshow(r)
plt.show()