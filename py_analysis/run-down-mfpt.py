
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl
from scipy import integrate
from scipy import optimize

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
def tdown(X): 
    return (X)*(1.0-ETSWUP(X)-(VARTSWUP(X)/(ETSWUP(X))))   #(X)*(ETSWUP(X)-ETSWUP2(X))/(ETSWUP(X))
def detODE(X):
    return numpy.array([sp.N(tup(X)-tdown(X))],dtype=float)
def phi_int(y): 
    return np.prod([tdown(float(zz)/NA)/tup(float(zz)/NA) for zz in range(1,y+1)])

def intexp2(x): 
    intexp = lambda y: 1.0/(tup(y)*phix(y))
    ph, err = integrate.quad(intexp,0,x)
    return ph

def phi(x): 
    N21 = 0.5/int(NA)
    intexp = lambda y: m.log(tup(y)/tdown(y))
    if x<N21:
        return 1
    ph, err = integrate.quad(intexp,N21,x+N21)
    return -ph
    
def phix(x): 
    N21 = 0.5/int(NA)
    intexp = lambda y: m.log(tup(y)/tdown(y))
    if x<N21:
        return 1
    ph, err = integrate.quad(intexp,N21,x+N21)
    return m.exp(-ph*NA)

numA = 21
alphas = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)


for acount in range(0,numA):
    wg = 0.35 - 0.0125 * float(acount)
    
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

    intexp1 = lambda y: phix(y)*intexp2(y)
   
    atimes[acount] = times[0]
    atimes2[acount] = sum([phix(i)*sum([1.0/(tup(float(z)/NA)*phix(z))  for z in range(0,i+1)])  for i in range(0,N2)] ) 
    #atimes[acount] = sum([phix(float(i)/NA)*sum([1.0/(tup(float(z)/NA)*phi(z))  for z in range(0,i+1)])  for i in range(0,N2)] ) 
    #ph, err = integrate.quad(intexp1,0,0.5)
#    atimes[acount] = ph
    x1=optimize.fsolve(detODE,0)[0]
    x3=optimize.fsolve(detODE,0.5)[0]
    
    dx=0.05
    dx12 =(phi(x1+ dx)-2*phi(x1)+phi(x1-dx))/(dx**2);
    dx32 =(phi(x3+ dx)-2*phi(x3)+phi(x3-dx))/(dx**2);
    #atimes2[acount] = (tup(x1)**-1*(abs(dx12*dx32))**-0.5)*3.142*m.exp(NA*(phi(x3)-phi(x1)))

    alphas[acount] = wg

plt.plot(alphas,atimes)
plt.yscale('log')
for a in range(np.size(atimes)): print alphas[a], atimes[a], atimes2[a]
