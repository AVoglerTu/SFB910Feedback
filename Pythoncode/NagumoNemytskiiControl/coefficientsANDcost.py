'''
Created on 11.04.2022

@author: alexa
'''

import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse import spdiags
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from stateCoupleduncontrolled import solveUc

#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- Global Parameter -------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#Terminal time
T=100

#Space 
a=20
    
#parameter for time discretization
dt=1/20
    
#parameter for dicretization of space
dx=1/20
    
nt=int(T/float(dt))
nx=int(a/float(dx))


#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- Parameter for equations ------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#dv=Dy Delta v+f(v)-eta*w+eps dW
#dw=Dz Delta w +gamma v-delta w

Dy=1
Dz=1

eta=0

delta=0.005
gamma=0.008


eps=dx

#boundary condition: bd=1:Neumann, bd=0: Dirichlet
bd=1


#initial conditions
y0=np.zeros([nx])

for k in range(nx):
    
    if k>=round(nx/float(4)) and k<=round(3*nx/float(4)):
        y0[k]=1


z0=np.zeros([nx])

#def u(x):
#    return x*(1-x)


#c=np.arange(0,a,dx)
#y0=u(c)

#plt.plot(y0)
#plt.show()

eps2=0.0000000000001

#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- radial basis -----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#number of basis functions for radial basis approximation
M=40


#parameter for exponential basis exp(-kappa*||y-z||^2)
kappa=6


#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- parameter for control problem ------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


#ga=1/float(10000)
ga=1/2

#running cost reference profile
w=np.zeros([nx,nt])
yref=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)



#yref=np.zeros([nx,nt])

#for k in range(nt):
#    yref[:,k]=y0

#terminal cost reference profile
yT=yref[:,nt-1]


#initial control
alpha2=0*np.ones([nx,M,nt])

#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- coefficients -----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#coefficient for non-linearity f
q=0.5

#define Nemytskii for non-linearity
def f(y):
    c=np.zeros([nx])
    for k in range(nx):
        c[k]=y[k]*(1-y[k])*(y[k]-q)
    return c

#def f(y):
#    c=np.zeros([nx])
#    return c

#define derivative for non-linearity
def df(y,v):
    c=np.zeros([nx])
    for k in range(nx):
        c[k]=(q*(2*y[k]-1)+(2-3*y[k])*y[k])*v[k]
    return c
 
#def df(y,v):
#    c=np.zeros([nx])
#    return c

#approximation of the control !(alpha2,z) are the control parameter
def b(y,alpha2,z):
    
    beta=np.zeros([nx])
    
    
    for k in range(M):
        beta=beta+np.multiply(alpha2[k,:],np.exp(-kappa*np.power(y-z[k],2)))

    return beta

#derivative for approximation control and corresponding adjoint operators
def db(y,alpha2,z,v):
    
    c=[]
    
    #directional derivative in direction v
    beta=np.zeros([nx])
    
    #adjoints for derivative
    g=np.zeros([M,nx])
    g1=np.zeros([M,1])
    g2=np.zeros([M])
    g3=np.zeros([nx])
    
    
    for k in range(M):
        
        #save radial basis in g for later use
        g[k,:]=np.exp(-kappa*np.power(y-z[k],2))
        
        grad=np.multiply(g[k,:],kappa*(y-z[k]))
        
        #calculate directional derivative in x in direction v
        beta=beta-2*np.multiply(alpha2[k,:],np.multiply(grad,v))
        
        #calculate adjoint of derivaives for gradient representation
        
        #adjoint for derivative in z evaluated at v
        g2[k]=2*np.multiply(v,alpha2[k,:]).dot(grad)*dx
        
        #print('v')
        #print(v)
        #print('alph')
        #print(alpha2[k,:])
        #print(np.multiply(v,alpha2[k,:]))
        
        #adjoint for derivative in x evaluated at v
        g3=g3+2*np.multiply(np.multiply(v,alpha2[k,:]),-grad)
    
    
    
    
    g1=np.multiply(g,v)
  
    #print(v.shape) 
    #print('g')
    #print(g)
    #print('v')
    #print(v)
    #print('g1')
    #print(g1)
  
    
    c.append(beta)
    c.append(g1)
    c.append(g2)
    c.append(g3)

    
        
    return c


    

#running cost coefficient
def l(t,y,yref,alpha2,z):

    
    return np.power(y-yref,2)+ga*np.power(b(y,alpha2,z),2)

#derivative running cost coefficient
def dl(t,y,yref,alpha2,z):
    
    return 2*(y-yref)+ga*2*db(y,alpha2,z,b(y,alpha2,z))[3]



#running cost
def lR(t,y,yref,alpha2,z):

    
    return dx*np.power(LA.norm(y-yref),2)+ga*dx*np.power(LA.norm(b(y,alpha2,z)),2)
    
#Terminal cost   
#def h(y,yT):
    
    
#    return dx*np.power(LA.norm(y-yT),2)

def h(y,yT):
    return np.power(LA.norm(y-yT),2)
    
#def dh(y,yT):
    
#    c=np.zeros([nx])
    
#    return 2*(y-yT)

def dh(y,yT):
    
    
    return 2*(y-yT)

#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- cost functional --------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
    

def cost(sol,alpha,yref,yT,z):
    
    cost0=0
    
    for k in range(nt):
        cost0=cost0+dt*lR(k*dt,sol[:,k],yref[:,k],alpha[:,:,k],z)
        
    
    cost0=cost0+dx*h(sol[:,nt-1],yT)
    
    return cost0



#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- cost functional --------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#number of Monte-Carlo approximations
N=1

#running deterministic or stochastic algorithm
det=0






