# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:06:18 2017

@author: sajid
"""

import numpy as np

def rect(x,d=0.5):
    if d!=0.5 :
        a = d
    else :
        a = 0.5
    y = np.zeros(np.shape(x))
    z = np.where(np.absolute(x)<a)
    y[z] = 1
    return y