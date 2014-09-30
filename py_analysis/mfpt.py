
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl

K = 80 
wg = 0.2875
ws = 0.514
alpha = 0.0
NA = 64

def psw( j ):
    gc = np.log(ws/(1-ws))*(K-2*j)/(4*wg)
    return 0.5 + 0.5*m.erf(m.sqrt(wg)*(1.0-gc))

def tup2( xx ):
    return psw(K*xx)

def tdown2( xx ):
    return (1.0 -  psw(K*xx))

def pxbar( i, nup, bp ):
    return sp.binomial(nup-1,i) * bp**i* (1-bp)**(nup-1-i)* tup2(i/float(Ns)) / sum(sp.binomial(nup-1,j) * bp**j* (1-bp)**(nup-1-j)* tup2(j/float(Ns)) for j in xrange(0,nup) )
 

# Function definition is here
def tup( X ):
    fX = int (X * NA)
    return (1-X)*sum(sp.binomial(fX,j) * bp**j * (1-bp)**(fX-j) * tup2(j/float(Ns)) for j in xrange(0,fX+1))

def tdown( X ):
    fX = int( X * NA)
    return (X)*sum(pxbar(j, fX,bp) * (tdown2(j/float(Ns))) for j in xrange(0,fX))


numA = 11
alphas = np.zeros(numA)
atimes = np.zeros(numA)

numX = 64
for acount in range(0,numA):
    alpha = 0.0#float(acount)/60.0
    wg = 0.5 - 0.025 * float(acount)
    #wg = 0.325
    L =  0.15 #float(ll)*0.2
    Ns = 32#min(64.0,2.0*m.floor(64.0*L))
    
    bp = Ns/float(numX)    
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
    
    fX=16
    X = fX/float(numX)
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