
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
Ns = 28
K = 90       
ws = 0.514

def tswup( xx ):
    gc = np.log(K)*sigma*(1.0-2.0*xx)/(2.0)
    if gc>1:
        return 0.5*m.exp(-(gc-1.0)/sigma)
    else:
        return 1.0 - 0.5*m.exp((gc-1.0)/sigma)

def ETSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
def RHO(X):  
    ap = 1.0-alpha
    PAB = (1.0/ETSWUP(X))*sum(sp.binomial(Ns,j) *(j/float(Ns))* X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
    PABL = ap * PAB + (1.0-ap)*X
    return (PABL-X)/(1.0-X)
def RHO2(X):  

    ap = 1.0-alpha
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    
   

    
    return ap* ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    return (O0 * X + (O1+O15)* X * (1-X) /float(Ns))/ (O0 + O2* X * (1-X) /float(Ns))#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    return PAB
def ETSWUP2(X):  

    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    rh =  alpha * ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )
    KK =   0.5*B**2 * X * (1-X) / float(Ns)

    sigma_2 = rh + (1.0-rh)/float(Ns)
    return 0.5*m.exp(-(A-B*X))*(1.0 + KK*(1.0 + ((alpha*B*X/(1+KK)) * (1.0- (1.0/float(Ns))))))
    return 0.5*m.exp(-(A-B*X))*(1.0 + 0.5*B**2* X * (1-X) * sigma_2)

def X1():  
#rh = RHO(X)

    rh = 0.0
    sigma_2 = rh + (1.0-rh)/float(Ns)
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    return 0.5*( m.exp(-(A))  ) / ( 1.0 -  0.5*( m.exp(-(A))  ) *(B +  0.5*B**2*   sigma_2) ) 
def X2(X):  
#rh = RHO(X)
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    VR = B**2*X*(1-X)/float(Ns)
    rh =  alpha * ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )
    KK =   0.5*B**2 * X * (1-X) / float(Ns)
    sigma_2 = rh + (1.0-rh)/float(Ns)
    return 0.5*m.exp(-(A-B*X))*(1.0 +  VR + ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return 0.5*m.exp(-(A-B*X))*(1.0 +  (0.5*B**2 * X * (1-X) / float(Ns))*(1.0 + ((alpha*B*X/(1+ (0.5*B**2 * X * (1-X) / float(Ns)))) * (1.0- (1.0/float(Ns))))))

    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    rh =  alpha * ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    sigma_2 = rh + (1.0-rh)/float(Ns)
    return 0.5*(1+(B*X-A) )*(1.0 + 0.5*B**2* X * (1-X) * sigma_2)
   
def ETSWUP3(X):  
    rh = RHO(X)

    rh = 0.0
    sigma = rh + (1.0-rh)/float(Ns)
    NSS = int(round(1.0/sigma))
 #   return sum(sp.binomial(NSS,j) * X**j* (1-X)**(NSS-j)* tswup(j/float(NSS)) for j in xrange(0,NSS+1))
    A = m.sqrt(wg) - m.sqrt(wg)*np.log(ws/(1-ws))*K*(1.0)/(4*wg)
    B =  m.sqrt(wg)*np.log(ws/(1-ws))*K/(2.0*wg)
    O0 = 0.0*(0.5 + 0.5*m.erf((A+B*X)))
    O2 = -(B**2/m.sqrt(m.pi))*(A+B*X)*m.exp(-(A+B*X)**2)
    O2 = -(B**2/m.sqrt(m.pi))*m.exp(-(A+B*X)**2)
    return O0 + O2* (A*X + (B-A)*X**2 - B*X**3)*sigma#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    
def VARTSWUP(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns))**2 for j in xrange(0,Ns+1))    - ETSWUP(X)**2
def tup(X):
    #return 0.5
    return (1-X)*(ETSWUP(X))
def tdown(X): 
    #return 0.5
    return (X)*(1.0-ETSWUP(X))
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
    sigma  = 130.0 #- 0.0125 * float(acount)
    wg  = 0.3 #- 0.0125 * float(acount)
    alpha = 0.4# + acount/float(numA-1)
    alphas[acount]=alpha
    
    N2 = int(NA*0.5+1)
#    print "====================="
#    print alpha
#    print "====================="
#    for x in range(0,NA):
#        print ETSWUP2(x/float(NA)), X2(x/float(NA))
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup(0)-tdown(0)
    Q[0,1]=tup(0)+tdown(0)

    Q[N2-1,N2-2]=tdown(N2/float(NA))
    Q[N2-1,N2-1]=1.0-tup(N2/float(NA))-tdown(N2/float(NA))
    for x in range(1,N2-1):
        Q[x,x-1] = tdown(x/float(NA))
        Q[x,x] = 1.0 - tup(x/float(NA)) - tdown(x/float(NA))
        Q[x,x+1] = tup(x/float(NA))
    
 

#    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
#    times = bb*np.matrix(np.ones((N2,1)))

   
#    atimes[acount] = times[0]
print ETSWUP2(0.1)

#aa= np.arange(65)/float(2*NA)
#for a in range(np.size(atimes)): print alphas[a], atimes[a]
#plt.plot(aa, [RHO(i) for i in aa])
#plt.plot(aa, [RHO2(i) for i in aa])
#plt.show()
