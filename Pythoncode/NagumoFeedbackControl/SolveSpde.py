'''
Created on 03.11.2022

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
from GalerkinSetting import laplace
import tensorflow as tf
from CoefficientsAndCost import *




#laplacian for given boundary conditions
A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))

def solveState(u0,model,w,solMin,solMax,normal):
    
    
    #initialize
    u=np.zeros([nt,K])

    u[0,:]=u0
    
    ypre=u0
    

    for r in range(1,nt):

        
        #compute non-linearity for current time step
        nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
    
        
        fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
        
        #evaluate stochastic control at current time step
        r1=tf.cast(ti[r-1], tf.double) 

        x1=tf.Variable([scale2*np.multiply(ypre-solMin,normal)])

        r1=tf.Variable([[scale*r1]])
        
        #print(tf.concat([r1,x1], axis=1))
        
        control=tf.reshape(model(tf.concat([r1,x1], axis=1)),[K])
        
        y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[r,:]+control)
        
        
        u[r,:]=y
        ypre=y

    return u

#uncontrolled solution
def solveStateUc(u0,w):
    
    
    #initialize
    u=np.zeros([nt,K])

    u[0,:]=u0
    
    ypre=u0
    

    for r in range(1,nt):

        
        #compute non-linearity for current time step
        nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
    
        
        fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
        
        
        y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[r,:])
        
        
        u[r,:]=y
        ypre=y
        
    return u
