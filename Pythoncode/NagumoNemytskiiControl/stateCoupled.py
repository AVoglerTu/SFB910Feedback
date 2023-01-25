'''
Created on 25.08.2018

@author: Alex
'''


from coefficientsANDcost import f, b
from functions import matM, matK, sparseDiag
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse import spdiags
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


#y0 initial solution, u control, params parameter for non-linearity
#dx and dy = space discretization
#dt= time discretization
#a=boundary for space

#projection of non-linearity onto finite dimensional subspace coefficients


def solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,alpha2,Dy,Dz,rad):
    
    nt=int(T/float(dt))
    nx=int(a/float(dx))
    
    M=matM(nx,dx,bd)
    K=matK(nx,dx,bd)
    
    
    #solution vector, where nt is time dimension and nx is the space dimension
    y=np.zeros([nx,nt])
    z=np.zeros([nx,nt])
    
    #initialize for time 0
    y[:,0]=y0
    z[:,0]=z0
    
    
    ypre=y0
    zpre=z0
    
    #initialize matrix for linear equation
    Mk=((1/dt)*M+Dy*K)
    Mk2=((1/dt+delta)*M+Dz*K)
    Mk=sparse.csr_matrix(Mk)
    Mk2=sparse.csr_matrix(Mk2)
    
    
    #calculate the solution at any time r
    for r in range(1,nt):
        
        
        #define feedback control u(t,x,w,a)=<a(t,x),phi(w)(x)>=<a(t,x),<b(w),psi(x)>>
        u=b(ypre,alpha2[:,:,r],rad)

        
        #solve the linear equation for the semi-implicit-Euler scheme
        ykt=spsolve(Mk,M.dot(f(ypre)-eta*zpre+(1/dt)*ypre+u)+(1/dt)*w[:,r])
        zkt=spsolve(Mk2,M.dot(gamma*ykt+(1/dt)*zpre))
            
        ypre=ykt
        zpre=zkt
        
        y[:,r]=ykt
        z[:,r]=zkt
    
    return y






