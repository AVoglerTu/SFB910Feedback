'''
Created on 03.11.2022

@author: alexa
'''
import numpy as np
import scipy


#boundary condition
bd=1

#space right interval bound
L=20

#number of Galerkin finite elements
K=400

#terminal time
T=20

#time discretization
dt=1/float(20)


nt=int(T/float(dt))

q=0.5

kappa=1

#space grid for fourier approximation
a=np.arange(0,L,L/float(K))

nx=a.size

#noise parameter 
eps=1/float(L)

#gradient norm threshold
gamma=0.00001

#normalizing time 
scale=1/float(100)

#normalizing space
scale2=1

#time interval
ti=np.arange(0,T,dt)


#initial condition
def uinit(y):
    
    nx=y.shape[0]
    
    u=np.zeros([nx])
    
    for i in range(nx):
        if 5<=y[i]<=15:
            u[i]=1
        
    return u

#define Nemytskii for non-linearity
def f(y):
    return 0

#derivative of Nemytskii non-linearity
def df(y):
    return 0

#terminal cost
def h(y,yT):
    return (1/float(2))*y**2

#derivative terminal cost
def dh(y,yT):
    
    return y

#running cost
def l(r,y,yref):
    return (1/float(2))*y**2

#derivative running cost
def dl(r,y,yref):
    return y

#define Hamiltonian of the system
#def H(r,y,p,alpha):
#    return tf.linalg.matmul()
