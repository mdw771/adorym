import numpy as np
import dxchange
import pandas as pd
import datetime
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('filename', default='None')
args = parser.parse_args()

filename = args.filename
filename_base = os.path.splitext(filename)[0]

f_orig = open(filename)
arr = []
for line in f_orig.readlines():
    line = line.replace(' (', '')
    line = line.replace(')', '')
    line = line.split(',')
    line[-1] = line[-1][:-2]
    this_line = []
    for d in line:
        this_real, this_imag = re.findall(r'-?\d+\.\d+e[+-]\d+', d)
        this_line.append(float(this_real) + 1j * float(this_imag))
    arr.append(np.array(this_line))
arr = np.stack(arr)

dxchange.write_tiff(abs(arr), filename_base + '_mag', dtype='float32', overwrite=True)
dxchange.write_tiff(np.angle(arr), filename_base + '_phase', dtype='float32', overwrite=True)