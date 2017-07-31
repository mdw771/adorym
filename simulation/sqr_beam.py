# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:24:04 2017

@author: sajid
"""

import numpy as np
import matplotlib.pyplot as plt
import rect as r
import prop as prop
L1 = 0.5 #side length
M = 250 #number of samples
step1 = L1/M #step size
x1 = np.linspace(-L1/2,L1/2,M) #input support coordinates
y1 = x1
wavel = 0.5*10**(-6) # wavelength of light
k = 2*np.pi/wavel #wavevector
w = 0.011 #width of square aperture
z = 2000 # propogation distance

X1,Y1 = np.meshgrid(x1,y1)
u1 = np.multiply(r.rect(X1/(2*w)),r.rect(Y1/(2*w))) #creating the input beam profile
I1 = abs(np.multiply(u1,u1))  #input intensity profile

ua = prop.propTF(u1,step1,L1,wavel,z) #result using TF method
Ia = abs(np.multiply(ua,ua)) #TF output intensity profile

ub = prop.propIR(u1,step1,L1,wavel,z) #result using IR method
Ib = abs(np.multiply(ub,ub)) #IR output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.subplot(121)
plt.imshow(abs(Ia),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('TF')
plt.subplot(122)
plt.imshow(abs(Ib),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('IR')
plt.show()