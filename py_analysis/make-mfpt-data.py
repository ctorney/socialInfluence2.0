
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
def VARTSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns))**2 for j in xrange(0,Ns+1))    - ETSWUP(X)**2
    
def tup(X):  
    return (1-X)*(ETSWUP(X)-(VARTSWUP(X)/(1.0-ETSWUP(X))))    

def tdown(X): 
    return (X)*(1.0-ETSWUP(X)-(VARTSWUP(X)/(ETSWUP(X))))   #(X)*(ETSWUP(X)-ETSWUP2(X))/(ETSWUP(X))
   
numA = 20
maxG = 0.46
minG = 0.24
dG = (maxG-minG)/numA
alphas = np.zeros(numA)
atimes = np.zeros(numA)

numX = 64
for Ns in range(8,32,4):
    acount = 0
    output = np.zeros((np.size(arange(minG,maxG,dG)),2))
    for wg in arange(minG,maxG,dG):
    
    
    
    
    
    
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
    
 

 
    
   
    #print (1-X)*sum(sp.binomial(fX,j) * bp**j * (1-bp)**(fX-j) * tup2(j/float(Ns)) for j in xrange(0,fX+1)), tup(X)
        bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    #for x in range(NA):
     #   print tup(x/float(NA)), tdown(x/float(NA))
    
        times = bb*np.matrix(np.ones((N2,1)))
        

        output[acount,1] = times[0]
        output[acount,0] = wg
        acount+=1
    np.save('an-time-' + str(Ns) + '.npy',output)
#plt.plot(alphas,atimes)
#for a in range(np.size(atimes)): print atimes[a], alphas[a]
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