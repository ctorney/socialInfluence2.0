
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
N2 = 32

def tswup( xx ):
    gc = np.log(ws/(1-ws))*K*(1.0-2.0*xx)/(4*wg)
    return 0.5 + 0.5*m.erf(m.sqrt(wg)*(1.0-gc))
def tswup2( xx ):
    B = (0.5) - (2.0*wg)/(np.log(ws/(1-ws))*K)
    A =  np.log(ws/(1-ws))*K/(2.0*m.sqrt(wg))
    return 0.5 + 0.5*m.erf(A*(xx-B))
def ttswup( Y, X ):
    A = m.sqrt(wg) - m.sqrt(wg)*np.log(ws/(1-ws))*K*(1.0)/(4*wg)
    B =  m.sqrt(wg)*np.log(ws/(1-ws))*K/(2.0*wg)
    O0 =  0.5 + 0.5*m.erf((A+B*X))
    O1 = (Y-X)*(B/m.sqrt(m.pi))*m.exp(-(A+B*X)**2)
    O2 = -0.5*(Y-X)**2*(B**2/m.sqrt(m.pi))*(A+B*X)*m.exp(-(A+B*X)**2)
    O3 = -(1.0/6.0)*(Y-X)**3*(2.0*B**3/m.sqrt(m.pi))*(1.0-2.0*(A+B*X)**2)*m.exp(-(A+B*X)**2)
    return O2

def ETSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
def ETSWUP2(X):  
    A = m.sqrt(wg) - m.sqrt(wg)*np.log(ws/(1-ws))*K*(1.0)/(4*wg)
    B =  m.sqrt(wg)*np.log(ws/(1-ws))*K/(2.0*wg)
    O0 =  0.5 + 0.5*m.erf((A+B*X))
    O1 = 0.0*(B/m.sqrt(m.pi))*m.exp(-(A+B*X)**2)
    O2 = -((B**2)/m.sqrt(m.pi))*(A+B*X)*m.exp(-(A+B*X)**2)
    O3 = 0*(B**3/(3*m.sqrt(m.pi)))*(2*(A+B*X)**2-1)*m.exp(-(A+B*X)**2)
    return O0 + O2* X * (1-X)/Ns + O3*X*(1-X)*(1-2*X)/Ns**2#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    return tswup(X) + X * m.sqrt(wg)*(np.log(ws/(1-ws))*K*(2.0)/(4*wg)) * m.exp((m.sqrt(wg)*(1.0-np.log(ws/(1-ws))*K*(1.0-2.0*X)/(4*wg)))**2)
numA = 1
alphas = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)

def ETSWUP3(X):  
    return (1.0/ETSWUP(X))*sum(sp.binomial(Ns,j) *(j/float(Ns))* X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))

for acount in range(0,numA):
    wg = 0.3 - 0.0125 * float(acount)
    
    for x in range(64):
        X=x/float(NA)
        print (ETSWUP(x/float(NA))), ETSWUP2(x/float(NA))
        #print (tswup2(x/float(NA))), tswup(x/float(NA))
    
 

