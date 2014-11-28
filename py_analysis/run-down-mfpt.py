
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl

NA = 64
Ns = 8
K = 80       
ws = 0.514

def tswup( xx ):
    gc = np.log(ws/(1-ws))*K*(1.0-2.0*xx)/(4*wg)
    return 0.5 + 0.5*m.erf(m.sqrt(wg)*(1.0-gc))

def ETSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
def ETSWUP2(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns))**2 for j in xrange(0,Ns+1))
def VARTSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns))**2 for j in xrange(0,Ns+1))    - ETSWUP(X)**2
    
def tup(X):  
    return (1-X)*(ETSWUP(X)-(VARTSWUP(X)/(1.0-ETSWUP(X))))    
    return (1-X)*(ETSWUP(X)-ETSWUP2(X))/(1.0-ETSWUP(X))
def tdown(X): 
    return (X)*(1.0-ETSWUP(X)-(VARTSWUP(X)/(ETSWUP(X))))   #(X)*(ETSWUP(X)-ETSWUP2(X))/(ETSWUP(X))
    

numA = 21
alphas = np.zeros(numA)
atimes = np.zeros(numA)


for acount in range(0,numA):
    wg = 0.5 - 0.0125 * float(acount)
    
    N2 = int(NA*0.5+1)
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup(0)-tdown(0)
    Q[0,1]=tup(0)

    Q[N2-1,N2-2]=tdown(N2/float(NA))
    Q[N2-1,N2-1]=1.0-tup(N2/float(NA))-tdown(N2/float(NA))
    for x in range(1,N2-1):
        Q[x,x-1] = tdown(x/float(NA))
        Q[x,x] = 1.0 - tup(x/float(NA)) - tdown(x/float(NA))
        Q[x,x+1] = tup(x/float(NA))
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))


    atimes[acount] = times[0]
    alphas[acount] = wg

plt.plot(alphas,atimes)
plt.yscale('log')
for a in range(np.size(atimes)): print alphas[a], atimes[a]