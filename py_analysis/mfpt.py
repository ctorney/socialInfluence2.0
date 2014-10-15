
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl

K = 80 
wg = 0.2875
ws = 0.505
alpha = 0.0
NA = 64

K = 80       
wg = 0.275
ws = 0.514

def psw( j ):
    gc = np.log(ws/(1-ws))*(K-2*j)/(4*wg)
    return 0.5 + 0.5*m.erf(m.sqrt(wg)*(1.0-gc))



def pxbar( i, nup, bp ):
    bp2 = float(nup)/float(numX)
    return sp.binomial(Ns,i) * bp2**i* (1-bp2)**(Ns-i)* tup(i/float(Ns)) / sum(sp.binomial(Ns,j) * bp2**j* (1-bp2)**(Ns-j)* tup(j/float(Ns)) for j in xrange(0,Ns+1) )
    


def pxbar2( i, nup, bp ):
    bp2 = float(nup)/float(numX)
    if (nup==0):    
        if (i==0):
            return 1
        else:
            return 0
    return sp.binomial(Ns,i) * bp2**i* (1-bp2)**(Ns-i)* tdown(i/float(Ns)) / sum(sp.binomial(Ns,j) * bp2**j* (1-bp2)**(Ns-j)* tdown(j/float(Ns)) for j in xrange(0,Ns+1) )
    

def tup( xx ):
    if xx>1.0: xx=1.0
    return psw(K*xx)

def tdown( xx ):
    if xx>1.0: xx=1.0
    return (1.0 -  psw(K*xx))
    
def tup2(X):  
    fX = int(X*NA)
    return (1-X)*sum(pxbar2(j, fX,bp) * (tup(j/float(Ns))) for j in xrange(0,Ns+1))
def tdown2(X): 
    fX = int(X*NA)
    if X==0:
        return 0
    return (X)*sum(pxbar(j, fX,bp) * (tdown(j/float(Ns))) for j in xrange(0,Ns+1))

numA = 21
alphas = np.zeros(numA)
atimes = np.zeros(numA)

numX = 64
for acount in range(0,numA):
    alpha = 0.0#float(acount)/60.0
    wg = 0.5 - 0.0125 * float(acount)
    #wg = 0.275
    L =  0.15 #float(ll)*0.2
    Ns = 8#min(64.0,2.0*m.floor(64.0*L))
    
    bp = Ns/float(numX)    
    N2 = int(NA*0.5+1)
     
   
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup2(0)-tdown2(0)
    Q[0,1]=tup2(0)

    Q[N2-1,N2-2]=tdown2(N2/float(NA))
    Q[N2-1,N2-1]=1.0-tup2(N2/float(NA))-tdown2(N2/float(NA))
    for x in range(1,N2-1):
        Q[x,x-1] = tdown2(x/float(NA))
        Q[x,x] = 1.0 - tup2(x/float(NA)) - tdown2(x/float(NA))
        Q[x,x+1] = tup2(x/float(NA))
    
 
    #print (1-X)*sum(sp.binomial(fX,j) * bp**j * (1-bp)**(fX-j) * tup2(j/float(Ns)) for j in xrange(0,fX+1)), tup(X)
    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    #for x in range(NA):
     #   print tup(x/float(NA)), tdown(x/float(NA))
    
    times = bb*np.matrix(np.ones((N2,1)))


    atimes[acount] = times[0]
    alphas[acount] = wg
plt.plot(alphas,atimes)
for a in range(np.size(atimes)): print atimes[a], alphas[a]
#NA = 34
#for acount in range(0,numA):
#    alpha = 0.0#float(acount)/60.0
#    wg = 0.28 + float(acount)/500.0
#    N2 = int(NA*0.5+1)
#    
#    Q = np.zeros((N2,N2))
#
#    Q[0,0]=1.0-tup(0)-tdown(0)
#    Q[0,1]=tup(0)
#
#    Q[N2-1,N2-2]=tdown(N2/float(NA))
#    Q[N2-1,N2-1]=1.0-tup(N2/float(NA))-tdown(N2/float(NA))
#    for x in range(1,N2-1):
#        Q[x,x-1] = tdown(x/float(NA))
#        Q[x,x] = 1.0 - tup(x/float(NA)) - tdown(x/float(NA))
#        Q[x,x+1] = tup(x/float(NA))
#    
#    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
#
#    
#    times = bb*np.matrix(np.ones((N2,1)))
#
#
#    atimes[acount] = times[0]
#    alphas[acount] = wg
#plt.plot(alphas,atimes)
plt.yscale('log')