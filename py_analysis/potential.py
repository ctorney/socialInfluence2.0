
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl

NA = 64
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
   
Xs = np.zeros(NA+1)
thGridup=np.zeros(NA+1)
thGriddown=np.zeros(NA+1)

transP = np.zeros((NA+1,2))

Ns=28
wg = 0.3
for fX in range(0,NA+1):#0,numX+1):
    X = fX/float(NA)
    Xs[fX] = X

    #thGridup[fX] = (1-X)*sum(sp.binomial(fX,j) * bp**j * (1-bp)**(fX-j) * tup(j/float(Ns)) for j in xrange(0,fX+1))
    thGridup[fX] = tup(X)#(1-X)*sum(pxbar2(j, fX,bp) * (tup(j/float(Ns))) for j in xrange(0,Ns+1))
    thGriddown[fX] = tdown(X)
    transP[fX,0] = thGridup[fX]
    transP[fX,1] = thGriddown[fX]
  
xGrid=np.arange(65)/64.0
plt.figure
for p in thGridup: print p
for p in thGriddown: print p

np.save('an-potential-' + str(Ns) + '.npy',transP)

plt.plot(Xs[0:32],thGridup[0:32],label='theory up')
plt.plot(Xs[0:32],thGriddown[0:32],label='theory up')
pot=np.log(np.divide(thGriddown,thGridup))
pot=np.cumsum(pot[1:64])
    
