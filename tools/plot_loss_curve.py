import matplotlib.pyplot as plt
import glob
import re
import pandas as pd
import numpy as np

flist = glob.glob('loss_rank_*.txt')
flist.sort()

d0 = pd.read_csv(flist[0])
n_batches = d0.shape[0]
n_ranks = len(flist)
loss_table = np.zeros([n_batches, n_ranks])

for f in flist:
    dset = pd.read_csv(f)
    rank = int(re.findall('\d+', f)[0])
    loss_table[:, rank] = dset['loss']

loss_table = np.mean(loss_table, axis=1)

plt.semilogy(loss_table)
plt.ylabel('Loss')
plt.xlabel('Batch index')
plt.savefig('loss.png',format='png')
