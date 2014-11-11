import numpy as np
import scipy
import matplotlib.pyplot as plt

    
def Init(num):
    return 2*np.pi*np.random.random_sample((num,))
    
# Metropolois Step 03
def YN(deltaE,T):
    if(deltaE < 0):
        return 1
    a = np.random.random()
    b = np.exp(-deltaE/T)
    if(a<b):
        return 1
    else:
        return 0

# Calculating < E >
def Energy(latt):
    length = len(latt)
    E = 0
    for x in range(length):
        for y in range(x+1, length):
            E -= 2*np.cos(latt[x] - latt[y])
    return E/length

def DeltaE(latt,xi,newElem):
    length = len(latt)
    dE = 0
    for i in range (length):
        if(i != xi):
            dE -= (2*np.cos(newElem-latt[i])-2*np.cos(latt[xi]-latt[i]))
    return dE

# Calculating Magnetiation
def Mag(latt):
    length = len(latt)
    mx = 0
    my = 0
    for i in range(length):
        mx += np.cos(latt[i])
        my += np.sin(latt[i])
    return (((mx)**2+(my)**2)**0.5)/length


def Newlatt(latt,T):
    length = len(latt)
    latt0 = []
    '''for i in range (length):
        latt0.append(latt[i])'''
    xi = np.random.randint(0,length-1)
    oldElem = latt[xi]
    newElem = np.random.random()*2*(np.pi)
    deltaE = DeltaE(latt,xi,newElem)
    M = YN(deltaE,T)
    if(M):
        latt[xi] = newElem
    return deltaE, latt

def Tempt(T,size):
    latt = Init(size)
    num = 500 #number of trials
    tempE = []
    tempm = []
    energy = Energy(latt)
    for p in range (num):
        deltaE, latt = Newlatt(latt,T)
        energy += deltaE
        if(p>=100):   
            tempE.append(energy)
            tempm.append(Mag(latt))
    avgE = np.mean(tempE)
    avgm = np.mean(tempm)
    stdevE = np.std(tempE)
    stdevm = np.std(tempm)
    return (avgE,avgm,stdevE,stdevm)
    

def main(size):
    q = 500
    E = []
    m = []
    Cv = []
    chiM = []

    J = -1.0/(2*q)
    z = q-1
    Tc = J*z
    #T = np.linspace(0.1,50,q)
    T = np.linspace(0.1*Tc,350*Tc,q)
    for a in range (0,q):
        (avgE,avgm,stdevE,stdevm) = Tempt(T[a],size)
        E.append(avgE)
        m.append(avgm)
        Cv.append(stdevE/T[a])
        chiM.append(stdevm/(T[a])**2)
    
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(T,E,'r')
    ax[0,1].plot(T,m,'g')
    ax[1,0].plot(T,Cv,'b')
    ax[1,0].set_xlim([0,3])
    ax[1,1].plot(T,chiM,'y')
    ax[1,1].set_xlim([0,3])
    plt.show()
    
if __name__=="__main__":
    main(16)
