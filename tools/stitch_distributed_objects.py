import dxchange
import numpy as np
import glob, os, re

flist_raw = glob.glob('*.tiff')
if 'delta' in flist_raw[0] or 'beta' in flist_raw[0]:
    type = 'delta_beta'
    name_part1 = 'delta'
    name_part2 = 'beta'
else:
    type = 'real_imag'
    name_part1 = 'mag'
    name_part2 = 'phase'

flist1 = []
flist2 = []
for f in flist_raw:
    if 'rank' in f:
        if 'delta' in f or 'mag' in f:
            flist1.append(f)
        elif 'beta' in f or 'phase' in f:
            flist2.append(f)

rank_ls = []
for f in flist1:
    rank_ls.append(int(re.findall('\d+', f)[-1]))
flist1 = np.array(flist1)[np.argsort(rank_ls)]
stack = []
for f in flist1:
    img = dxchange.read_tiff(f)
    stack.append(img)
stack = np.concatenate(stack, 0)
dxchange.write_tiff(stack, '{}_stack'.format(name_part1), overwrite=True, dtype='float32')

if len(flist2) > 0:
    rank_ls = []
    for f in flist2:
        rank_ls.append(int(re.findall('\d+', f)[-1]))
    flist2 = np.array(flist2)[np.argsort(rank_ls)]
stack = []
for f in flist2:
    img = dxchange.read_tiff(f)
    stack.append(img)
stack = np.concatenate(stack, 0)
dxchange.write_tiff(stack, '{}_stack'.format(name_part2), overwrite=True, dtype='float32')
