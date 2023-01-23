'''
Created on 08.11.2022

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

def adjoint(sol,model,yT,yref,solMin,solMax,normal):
    
    #initialize adjoint
    p=np.zeros([nt,K])
    
    #terminal condition
    p[nt-1,:]=dh(sol[nt-1,:],yT)
    
    ppre=p[nt-1,:]
    
    for r in range(nt-2,-1,-1):
        
        
        #recover solution and adjoint state from prev. time step
        rsol=np.sqrt(K/float(L))*scipy.fftpack.idct(sol[r+1,:],type=2, norm='ortho')
        rp=np.sqrt(K/float(L))*scipy.fftpack.idct(ppre,type=2, norm='ortho')
        
        #Nemytskii non-linearity evaluated
        nonlin1=np.multiply(rp,df(rsol))
        
       
        #Fourier coefficients of evaluated non-linearity
        dfFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin1,type=2, norm='ortho')
        
        #evaluate stochastic control at current time step
        r1=tf.cast(ti[r+1], tf.double) 
        y1=tf.cast(tf.reshape(tf.Variable(ppre),[K,1]),tf.double)
        x1=tf.Variable([scale2*np.multiply(sol[r+1,:]-solMin,normal)])
        r1=tf.Variable([[scale*r1]])
        
        #print(tf.concat([r1,x1], axis=1))
        
        with tf.GradientTape() as tape:
            
            tape.watch(x1)

            control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[1,K]),tf.double)
            
            J=tf.linalg.matmul(control,y1)
     
            gradients = tf.reshape(tape.gradient(J, x1),[K])

    
        y=spsolve(A,(1/float(dt))*ppre+dfFourier+gradients+dl(r+1,sol[r+1,:],yref))
    
        
        p[r,:]=y
        
        ppre=y
        
    return p



def grad(p,sol,model,yT,yref,solMin,solMax,normal):
    
    
    #normalize input values for neural network
    
    #cost
    c=0
    c1=0
    c2=0
    
    #vector for gradient
    g=0
    
    ou=[]
    
    with tf.GradientTape() as tape:
        
        #calculate int_0^T H_alpha(r,x_t,p_t,alpha_t) dt
        for r in range(nt):
            
            y1=tf.cast(tf.reshape(tf.Variable(p[r,:]),[K,1]),tf.double)
            x1=tf.Variable([scale2*np.multiply(sol[r,:]-solMin,normal)])
            r1=tf.Variable([[tf.cast(scale*ti[r],tf.double)]])
        
            #evaluate feedback control
            control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[1,K]),tf.double)
            
            #calculate reduced hamiltonian
            g=g+dt*tf.linalg.matmul(control,y1)+dt*tf.norm(control)**2
            
            #calculate cost
            c1=c1+dt*LA.norm(sol[r,:]-yref[r,:])**2
            c2=c2+dt*tf.norm(control)**2
            
        
        print('diffcost')
        print(c1)
        print('controlcost')
        print(c2) 
        c=c1+c2
    #derive gradient as derivative of hamiltonian i.e. d/d(alpha) int_0^T H_alpha(r,x_t,p_t,alpha_t) dt
    gradients=tape.gradient(g,model.trainable_variables)
    
    c=c+LA.norm(sol[nt-1,:]-yT)

    ou.append(gradients)
    ou.append(g)
    ou.append(c)
    return ou
