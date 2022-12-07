import adorym
import numpy as np
import torch
import matplotlib.pyplot as plt

from adorym.wrappers import fft2_and_shift

data = np.zeros((100,100))
data[30:40,30:40] = 1

data = data + 0j
shift = [10,10]

backend = 'pytorch'
data = adorym.create_variable(data, backend=backend)
data1 = adorym.norm_complex(data, backend=backend)

# if backend=='autograd':
#     data2 = adorym.ifft(data.real.detach().numpy(), data.imag.detach().numpy(), backend=backend)
# else:
data2 = adorym.norm(data.real, data.imag, backend=backend)

if backend == 'pytorch':
    data2 = [i.detach().numpy() for i in data2]
    data1 = data1.detach().numpy()

plt.figure(1, clear=True)
plt.subplot(321)
# plt.imshow(data2[0] - data1.real)
plt.imshow(data2 - data1)
plt.subplot(322)
# plt.imshow(data2[1] - data1.imag)
plt.imshow(data1)
plt.subplot(323)
# plt.imshow(data1.real)
plt.imshow(data2)
# plt.subplot(324)
# plt.imshow(data2[0])
# plt.subplot(325)
# plt.imshow(data1.imag)
# plt.subplot(326)
# plt.imshow(data2[1])

# functions checked with their _complex counterparts
# fft
# realign_image_fourier
# exp
# ifft
# fft2
# ifft2
# fft2_and_shift
# ifft2_and_shift
# ishift_and_ifft2
# convolve_with_transfer_function
# convovle_with_impulse_response
# norm

# all complex functions return identical outputs


