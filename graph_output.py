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
plt.subplot(2, 1, 1)
plt.plot(temp, nrg)
plt.subplot(2, 1, 2)
plt.plot(temp, mag)
plt.show()
