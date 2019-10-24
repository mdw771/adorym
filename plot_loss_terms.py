import numpy as np
import matplotlib.pyplot as plt


f = open('loss.txt', 'r')
lines = f.readlines()

alpha = []
total_loss = []
diff_loss = []
tv_loss = []

for line in lines:
    line = line.split(' ')
    alpha.append(float(line[0]))
    total_loss.append(float(line[1]))
    diff_loss.append(float(line[2]))
    tv_loss.append(float(line[3]))

plt.figure()
plt.semilogx(alpha, total_loss, label='Total')
plt.semilogx(alpha, diff_loss, label='Mismatch')
plt.semilogx(alpha, tv_loss, label='TV')
plt.legend()
# plt.show()
plt.savefig('loss.png')