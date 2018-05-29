import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

for ds in [4, 2, 1]:
    error = np.load('convergence/error_ds_{}.npy'.format(ds))
    plt.semilogy(error, '-o', label='Downsampling {}x'.format(ds))
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Data error')
plt.show()
