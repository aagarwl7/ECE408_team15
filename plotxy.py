import numpy as np
from scipy.optimize import newton
import scipy.special as sp
import matplotlib.pyplot as plt


N = 1000000
J = -1.0/(2*N)
z = N-1
Tc = 1.0*z*J
mag = [];E = [];chiM = [];Cv = [];beta = [];
    
p = 100
T = np.linspace(0.1,4,p)
#T = np.linspace(0.5*Tc,350*Tc,p)

for x in range (p):
    beta.append(1/T[x])
    
for x in range (p):
    func = lambda m : m+(sp.iv(1,2*z*J*beta[x]*m)/sp.iv(0,2*z*J*beta[x]*m))
    mag.append(newton(func,0.5))
    
for x in range (p):
    E.append(J*z*mag[x]**2)
    
for x in range (p):
    k = beta[x]*2*J*z*mag[x]
    chiM.append(beta[x]*((sp.iv(0,k)+sp.iv(2,k))/(2*sp.iv(0,k))-(sp.iv(1,k)/sp.iv(0,k))**2))
    
for x in range (p):
    Cv.append(((2*J*z*mag[x])**2)*beta[x]*chiM[x])

fig,ax = plt.subplots(2,2)

ax[0,0].set_xlabel('Temperature')
ax[0,0].set_ylabel('Energy')
ax[0,0].plot(T,E,'r')

ax[0,1].set_xlabel('Temperature')
ax[0,1].set_ylabel('Magnetization')
ax[0,1].plot(T,mag,'g')

ax[1,0].set_xlabel('Temperature')
ax[1,0].set_ylabel('Specific Heat')
ax[1,0].plot(T,Cv,'b')

ax[1,1].set_xlabel('Temperature')
ax[1,1].set_ylabel('Susceptibility')
ax[1,1].plot(T,chiM,'y')

plt.show()

                   
