
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
def RHO(X):  
    ap = 1.0-alpha
    PAB = (1.0/ETSWUP(X))*sum(sp.binomial(Ns,j) *(j/float(Ns))* X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
    PABL = 0.75*ap**2 * PAB + (1.0-0.75*ap**2)*X
    return (PABL-X)/(1.0-X)
def ETSWUP2(X):  
    rh = RHO(X)

 #   rh = 0.0
    sigma = rh + (1.0-rh)/float(Ns)
    NSS = int(round(1.0/sigma))
 #   return sum(sp.binomial(NSS,j) * X**j* (1-X)**(NSS-j)* tswup(j/float(NSS)) for j in xrange(0,NSS+1))
    A = m.sqrt(wg) - m.sqrt(wg)*np.log(ws/(1-ws))*K*(1.0)/(4*wg)
    B =  m.sqrt(wg)*np.log(ws/(1-ws))*K/(2.0*wg)
    O0 =  0.5 + 0.5*m.erf((A+B*X))
    O2 = -(B**2/m.sqrt(m.pi))*(A+B*X)*m.exp(-(A+B*X)**2)
    return O0 + O2* X * (1-X) * sigma#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    
def ETSWUP3(X):  
    rh = RHO(X)

    rh = 0.0
    sigma = rh + (1.0-rh)/float(Ns)
    NSS = int(round(1.0/sigma))
 #   return sum(sp.binomial(NSS,j) * X**j* (1-X)**(NSS-j)* tswup(j/float(NSS)) for j in xrange(0,NSS+1))
    A = m.sqrt(wg) - m.sqrt(wg)*np.log(ws/(1-ws))*K*(1.0)/(4*wg)
    B =  m.sqrt(wg)*np.log(ws/(1-ws))*K/(2.0*wg)
    O0 =  0.5 + 0.5*m.erf((A))
    O2 = -(B**2/m.sqrt(m.pi))*(A+B*X)*m.exp(-(A+B*X)**2)
    return O0 + X * (m.exp(-A**2)/m.sqrt(m.pi))*B*(1-A*B/Ns)
    return O0 + O2* X * (1-X) * sigma#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
def VARTSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns))**2 for j in xrange(0,Ns+1))    - ETSWUP(X)**2
def tup(X):
    #return 0.5
    return (1-X)*(ETSWUP3(X))
def tdown(X): 
    #return 0.5
    return (X)*(1.0-ETSWUP3(X))
def detODE(X):
    return np.array([sp.N(tup(X)-tdown(X))],dtype=float)
def phi_int(y): 
    return np.prod([tdown(float(zz)/NA)/tup(float(zz)/NA) for zz in range(1,y+1)])
def diffdown(x):
    h=0.001
    return (tdown(x+0.5*h)-tdown(x-0.5*h))/h

def diffup(x):
    h=0.000001
    return (tup(x+0.5*h)-tup(x-0.5*h))/h
    
def diff2phi(x):
    return -((1.0/tup(x))*diffup(x)-(1.0/tdown(x))*diffdown(x))
def intexp2(x): 
    intexp = lambda y: 1.0/(tup(y)*phix(y))
    ph, err = integrate.quad(intexp,0,x)
    return ph

def phi(x): 
    N21 = 0.5/int(NA)
    intexp = lambda y: m.log(tdown(y)/tup(y))
    if x<N21:
        return 0
    ph, err = integrate.quad(intexp,0,x)
    return ph
    
def phix(x): 
    N21 = 0.5/int(NA)
    print x
    intexp = lambda y: m.log(tup(y)/tdown(y))
    if x<N21:
        return 1
    ph, err = integrate.quad(intexp,N21,x+N21)
    return m.exp(-ph*NA)

numA = 1
alphas = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)


for acount in range(numA):
    wg = 0.3 #- 0.0125 * float(acount)
    alpha = 1.0# + acount/float(numA-1)
    alphas[acount]=alpha
    
    N2 = int(NA*0.5+1)
#    print "====================="
#    print alpha
#    print "====================="
    for x in range(0,NA):
        print tup(x/float(NA)), tdown(x/float(NA)), ETSWUP2(x/float(NA)), ETSWUP3(x/float(NA))
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup(0)-tdown(0)
    Q[0,1]=tup(0)+tdown(0)

    Q[N2-1,N2-2]=tdown(N2/float(NA))
    Q[N2-1,N2-1]=1.0-tup(N2/float(NA))-tdown(N2/float(NA))
    for x in range(1,N2-1):
        Q[x,x-1] = tdown(x/float(NA))
        Q[x,x] = 1.0 - tup(x/float(NA)) - tdown(x/float(NA))
        Q[x,x+1] = tup(x/float(NA))
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))

   
    atimes[acount] = times[0]
aa= np.arange(65)/float(NA)
for a in range(np.size(atimes)): print alphas[a], atimes[a]
#plt.plot(aa, [tup(i) for i in aa])
#plt.show()
