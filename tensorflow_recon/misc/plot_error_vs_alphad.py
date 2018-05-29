import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

alphad = np.arange(-8, -4)
alphad = np.power(10., alphad)
error = np.array([0.000150185931633,
                  0.000295155483641,
                  0.000145008280322,
                  0.000285498569913])

a = plt.loglog(alphad, error)
plt.xlabel(r'$\alpha_d$')
plt.ylabel('Mean squared error per voxel')
plt.grid(True)
plt.show()