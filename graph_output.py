import csv
import sys
from matplotlib import pyplot as plt

f = open(sys.argv[1], 'r')
r = csv.reader(f, delimiter=',')
data = [map(float, d) for d in list(r) if len(d)==3][1:]
temp = [x[0] for x in data]
nrg = [x[1] for x in data]
mag = [x[2] for x in data]

plt.figure(1)
ax = plt.subplot(1, 1, 1)
ax.set_title("1000 iterations on CPU")
ax.set_xlabel("Temperature")
ax1 = plt.subplot(2, 1, 1)
ax1.set_title("Energy")
ax1.plot(temp, nrg)
ax2 = plt.subplot(2, 1, 2)
ax2.set_title("Magnetization")
ax2.plot(temp, mag)
plt.show()

