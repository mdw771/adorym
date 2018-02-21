from xdesign import *
import numpy as np


silicon = XraylibMaterial('Si', 2.33)

ball0 = Sphere_3d(center=Point([20, 40, 55]), radius=7)
phantom = Phantom(geometry=ball0, material=silicon)

ball1 = Sphere_3d(center=Point([40, 20, 10]), radius=7)
ball1 = Phantom(geometry=ball1, material=silicon)

phantom.children.append(ball1)

grid = discrete_phantom(phantom, 1,
                        bounding_box=[[0, 64],
                                      [0, 64],
                                      [0, 64]],
                        prop=['delta', 'beta'],
                        ratio=1,
                        mkwargs={'energy': 5},
                        overlay_mode='replace')

grid_delta = grid[..., 0]
grid_beta = grid[..., 1]
print(grid_delta.shape)

np.save('dual_sphere_delta', grid_delta)
np.save('dual_sphere_beta', grid_beta)