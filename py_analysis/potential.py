
#!/usr/bin/python

import sympy as sp
import scipy as sc
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl

K = 80       
wg = 0.275
ws = 0.514

def psw( j ):
    gc = np.log(ws/(1-ws))*(K-2*j)/(4*wg)
    return 0.5 + 0.5*m.erf(m.sqrt(wg)*(1.0-gc))



def pxbar( i, nup, bp ):
    bp2 = float(nup-1)/float(numX)
    return sp.binomial(Ns,i) * bp2**i* (1-bp2)**(Ns-i)* tup(i/float(Ns)) / sum(sp.binomial(Ns,j) * bp2**j* (1-bp2)**(Ns-j)* tup(j/float(Ns)) for j in xrange(0,Ns+1) )
    return sp.binomial(nup-1,i) * bp**i* (1-bp)**(nup-1-i)* tup(i/float(Ns)) / sum(sp.binomial(nup-1,j) * bp**j* (1-bp)**(nup-1-j)* tup(j/float(Ns)) for j in xrange(0,nup) )


def pxbar2( i, nup, bp ):
    bp2 = float(nup)/float(numX)
    if (nup==0):    
        if (i==0):
            return 1
        else:
            return 0
    return sp.binomial(Ns,i) * bp2**i* (1-bp2)**(Ns-i)* tdown(i/float(Ns)) / sum(sp.binomial(Ns,j) * bp2**j* (1-bp2)**(Ns-j)* tdown(j/float(Ns)) for j in xrange(0,Ns+1) )
    return sp.binomial(nup,i) * bp**i* (1-bp)**(nup-i)* tdown(i/float(Ns)) / sum(sp.binomial(nup,j) * bp**j* (1-bp)**(nup-j)* tdown(j/float(Ns)) for j in xrange(0,nup+1) )

def tup( xx ):
    if xx>1.0: xx=1.0
    return psw(K*xx)

def tdown( xx ):
    if xx>1.0: xx=1.0
    return (1.0 -  psw(K*xx))
    
def tup2(X):  
    return (1-X)*sum(pxbar2(j, fX,bp) * (tup(j/float(Ns))) for j in xrange(0,Ns+1))
def tdown2(X): 
    return (X)*sum(pxbar(j, fX,bp) * (tdown(j/float(Ns))) for j in xrange(0,Ns+1))

numX = 64
Xs = np.zeros(numX+1)
thGridup=np.zeros(numX+1)
thGriddown=np.zeros(numX+1)

#a=np.loadtxt("/home/ctorney/data.txt")
#for p in np.unique(a[:,1]): print p, psw(K*np.mean(a[a[:,1]==p,2]))

L = 0.15

        
for ll in range(28,32,4):
    L =  0.5 #float(ll)*0.2
    Ns = ll * 2#min(64.0,2.0*m.floor(64.0*L))
    print Ns
    Ns=32
    
    bp = Ns/float(numX)
    #print bp
    for fX in range(0,58):#0,numX+1):
        X = fX/float(numX)
        Xs[fX] = X
    
        #thGridup[fX] = (1-X)*sum(sp.binomial(fX,j) * bp**j * (1-bp)**(fX-j) * tup(j/float(Ns)) for j in xrange(0,fX+1))
        thGridup[fX] = tup2(X)#(1-X)*sum(pxbar2(j, fX,bp) * (tup(j/float(Ns))) for j in xrange(0,Ns+1))
        thGriddown[fX] = tdown2(X)
        
  
    xGrid=np.arange(65)/64.0
    plt.figure
    for p in thGridup: print p
    for p in thGriddown: print p
    
    plt.plot(Xs[0:32],thGridup[0:32],label='theory up')
    plt.plot(Xs[0:32],thGriddown[0:32],label='theory up')
    pot=np.log(np.divide(thGriddown,thGridup))
    pot=np.cumsum(pot[1:64])
    #plt.plot(xGrid[1:32],pot[1:32],label = str(float(ll)/float(numX)))
    #plt.axis([0, 0.5, -4, 20])
    
