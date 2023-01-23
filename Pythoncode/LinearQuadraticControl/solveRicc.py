'''
Created on 05.12.2022

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
from test.test_itertools import isOdd
from scipy import integrate
from CoefficientsAndCost import *
from scipy import fftpack
from GalerkinSetting import laplace


def Ricc():
    
    V=np.zeros([nt,K,K])
    
    V[nt-1,:,:]=np.identity(K)
    
    for r in range(nt-2,-1,-1):
        
        print(r)
        
        V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+V[r+1,:,:]*(kappa)+np.transpose(laplace()),V[r+1,:,:]*(1/float(dt))+np.multiply(V[r+1,:,:],-laplace())+np.identity(K))

    return V

def Ricc2(V,yref):
    
    phi=np.zeros([nt,K])

    for r in range(nt-2,-1,-1):
        print(r)
        
        phi[r,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))-np.transpose(-laplace()-V[r+1,:,:]*(kappa)),phi[r+1,:]*(1/float(dt)))

    return phi


def RiccRef():
    
    V=np.zeros([nt,K,K])
    
    V[nt-1,:,:]=np.identity(K)
    
    for r in range(nt-2,-1,-1):
        
        print(r)
        
        V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+V[r+1,:,:]*(kappa)+np.transpose(laplace()),V[r+1,:,:]*(1/float(dt))+np.multiply(V[r+1,:,:],-laplace())+np.identity(K))

    return V

def RiccRef2(V,yref):
    
    phi=np.zeros([nt,K])

    for r in range(nt-2,-1,-1):
        print(r)
        
        phi[r,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))-np.transpose(-laplace()-V[r+1,:,:]*(kappa)),phi[r+1,:]*(1/float(dt))-yref[r+1,:])

    return phi




def solveStateRiccRef(u0,w,V,phi,yref,yT):
    
    A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))
    
    print(phi)
    
    out=[]
    
    #initialize
    u=np.zeros([nt,K])
    v=np.zeros([nt,K])

    u[0,:]=u0
    v[0,:]=-kappa*V[0,:,:].dot(u0)-kappa*phi[0,:]
    
    ypre=u0
    

    for r in range(1,nt):

        Vr=sparse.csr_matrix(kappa*V[r,:,:])
        
        y=spsolve(A+Vr,(1/float(dt))*ypre+(1/float(dt))*w[r,:]-kappa*phi[r,:])
        v[r,:]=-kappa*V[r,:,:].dot(y)-kappa*phi[r,:]
        
        u[r,:]=y
        ypre=y

    out.append(u)
    out.append(v)
    
    return out



def solveState0(u0,w,yref,yT):
    
    A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))

    

    
    #initialize
    u=np.zeros([nt,K])


    u[0,:]=u0

    
    ypre=u0
    

    for r in range(1,nt):

        
        y=spsolve(A,(1/float(dt))*ypre+(1/float(dt))*w[r,:])

        
        u[r,:]=y
        ypre=y
    
    return u




