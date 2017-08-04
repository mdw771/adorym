# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:38:01 2017

@author: sajid
"""

import numpy as np
import numpy.fft as fft

'''
Propogation using the Transfer function method. Note that fftfreq has been used from the numpy.fft library. Using this means that we no longer perform an fftshift after transforming u1 to frequency domain.

u1 is the profile of the beam at the input plane. 
step is the sampling step size at the input plane.
L is the side length of the support.
wavel is the wavelength of the light
z is the propogation distance

u2 is the beam profile at the output plane
'''
def propTF(u1,step,L,wavel,z) :
    M,N = np.shape(u1)
    #k = 2*np.pi/wavel
    fx = fft.fftfreq(M,d=step)
    fy = fft.fftfreq(N,d=step)
    FX,FY = np.meshgrid((fx),(fy))
    FX = fft.fftshift(FX)
    FY = fft.fftshift(FY)
    H = np.exp(-1j*np.pi*wavel*z*(FX**2+FY**2))
    U1 = fft.fftshift(fft.fft2(u1))
    U2 = np.multiply(H,(U1))
    u2 = fft.ifft2(fft.ifftshift(U2))
    return u2
'''
Propogation using the Impulse Response function. The convention of shiftinng a function in realspace before performing the fourier transform which is used in the reference is followed here. Input convention as above
'''
def propIR(u1,step,L,wavel,z):
    M,N = np.shape(u1)
    k = 2*np.pi/wavel
    x = np.linspace(-L/2.0,L/2.0-step,M)
    y = np.linspace(-L/2.0,L/2.0-step,N)
    X,Y = np.meshgrid(x,y)
    h = (np.exp(1j*k*z)/(1j*wavel*z))*np.exp(1j*k*(1./(2*z))*(X**2+Y**2))
    H = fft.fft2(fft.fftshift(h))*step*step
    U1 = fft.fft2(fft.fftshift(u1))
    U2 = H * U1
    u2 = fft.ifftshift(fft.ifft2(U2))
    return u2
'''
Fraunhofer propogation. Note that we now output two variables since the side length of the observation plane is no longer the same as the side length of the input plane.
'''
def propFF(u1,step,L1,wavel,z):
    M,N = np.shape(u1)
    k = 2*np.pi/wavel
    L2 = wavel*z/step
    step2 = wavel*z/L1
    n = L2/step2 #number of samples
    x2 = np.linspace(-L2/2.0,L2/2.0,n)
    X2,Y2 = np.meshgrid(x2,x2)
    c = 1/(1j*wavel*z)*np.exp(((1j*k)/(2.*z))*(X2**2+Y2**2))
    u2 = np.multiply(c,fft.ifftshift(fft.fft2(fft.fftshift(u1))))*step*step
    return u2,L2

def prop1FT(u1,step,L1,wavel,z):
    M,N = np.shape(u1)
    k = 2*np.pi/wavel
    print(wavel,z,step)
    x = np.linspace(-L1/2.0,L1/2.0-step,M)
    y = np.linspace(-L1/2.0,L1/2.0-step,N)
    X,Y = np.meshgrid(x,y)
    L2 = wavel*z/step
    step2 = wavel*z/L1
    n = L2/step2 #number of samples
    x2 = np.linspace(-L2/2.0,L2/2.0,n)
    X2,Y2 = np.meshgrid(x2,x2)
    c = 1/(1j*wavel*z)*np.exp(((1j*k)/(2.*z))*(X2**2+Y2**2))
    c0 = np.exp(1j * k / (2 * z) * (X**2 + Y**2))
    u2 = np.multiply(c,fft.ifftshift(fft.fft2(fft.fftshift(u1*c0))))*step*step
    return u2,L2