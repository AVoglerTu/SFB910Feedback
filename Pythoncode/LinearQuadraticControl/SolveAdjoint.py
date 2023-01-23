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
        
        
        #normalize input for neural network
        r1=tf.cast(ti[r]+1, tf.double) 
        y1=tf.cast(tf.reshape(tf.Variable(ppre),[K,1]),tf.double)
        x1=tf.Variable([scale2*np.multiply(sol[r+1,:]-solMin,normal)])
        r1=tf.Variable([[scale*r1]])

        #calculate H_x(t,x_t,p_t,alpha_t) 
        with tf.GradientTape() as tape:
            
            tape.watch(x1)
            
            #evaluate control
            control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[1,K]),tf.double)
            
            #calculate H(t,x_t,p_t,alpha_t)
            J=tf.linalg.matmul(control,y1)
            
            #differentiate
            gradients = tf.reshape(tape.gradient(J, x1),[K])

    
        y=spsolve(A,(1/float(dt))*ppre+gradients+dl(r+1,sol[r+1,:],yref))
    
        
        p[r,:]=y
        
        ppre=y
        
    return p



def grad(p,sol,model,yT,yref,solMin,solMax,normal):
    
    #cost
    c=0
    
    #gradient
    g=0
    
    #output list
    ou=[]
    
    with tf.GradientTape() as tape:
        
        #calculate int_0^T H_alpha(t,x_t,p_t,alpha_t) dt
        for r in range(nt):
            
            #normalize input for neural network
            y1=tf.cast(tf.reshape(tf.Variable(p[r,:]),[K,1]),tf.double)
            x1=tf.Variable([scale2*np.multiply(sol[r,:]-solMin,normal)])
            r1=tf.Variable([[tf.cast(scale*ti[r],tf.double)]])
        
            #evaluate feedback control
            control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[1,K]),tf.double)
            
            #calculate reduced hamiltonian
            g=g+dt*tf.linalg.matmul(control,y1)+(1/float(2))*dt*tf.norm(control)**2
            
            #calculate cost at time r
            c=c+dt*(1/float(2))*LA.norm(sol[r,:])**2+(1/float(2))*dt*tf.norm(control)**2
    
    #derive gradient as derivative of hamiltonian i.e. d/d(alpha) int_0^T H(r,x_t,p_t,alpha_t) dt
    gradients=tape.gradient(g,model.trainable_variables)
    
    #cost at terminal time
    c=c+(1/float(2))*LA.norm(sol[nt-1,:])**2

    ou.append(gradients)
    ou.append(g)
    ou.append(c)
    return ou
