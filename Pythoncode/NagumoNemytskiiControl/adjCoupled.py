'''
Created on 30.08.2018

@author: Alex
'''

from functions import matM, matK
from coefficientsANDcost import b, df, dh, dl, db, dh, ga
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse


def adjoint(L,T,a,y,yref,yT,dx,dt,bd,alpha,delta,gamma,eta,Dy,Dz,z):
    
    
    #output
    c=[]
    
    nt=int(T/float(dt))
    nx=int(a/float(dx))
    
    M=matM(nx,dx,bd)
    K=matK(nx,dx,bd)
    

    
    #initialize adjoint vector
    p1=np.zeros([nx,nt])
    p2=np.zeros([nx,nt])
    
    #initialize matrix for linear equation
    Mk=((1/dt)*M+Dy*K)
    Mk2=((1/dt+delta)*M+Dz*K)
    Mk=sparse.csr_matrix(Mk)
    Mk2=sparse.csr_matrix(Mk2)

    
    p1[:,nt-1]=dh(y[:,nt-1],yT)

    
    ppre1=p1[:,nt-1]
    #ppre2=p2[:,nt-1]
    
    #gradient coefficients
    g=np.zeros([L,nx,nt])
    
    #gradient nodes
    gz=np.zeros([L])
    
    
    g[:,:,nt-1]=db(y[:,nt-1], alpha[:,:,nt-1],z,ppre1)[1]+2*ga*db(y[:,nt-1], alpha[:,:,nt-1],z, b(y[:,nt-1],alpha[:,:,nt-1],z))[1]
    

    #calculate adjoint backward in time
    for k in range(nt-2,-1,-1):
        
        ytk=y[:,k]
        
        #evaluate gateaux differential of control in state in direction of the adjoint
        #dbx=db(ytk, alpha[:,:,k],z,ppre1)
        
        
        phi1=spsolve(Mk,M.dot(df(ytk, ppre1)+db(ytk, alpha[:,:,k],z,ppre1)[0]+dl(k*dt, ytk, yref[:,k], alpha[:,:,k],z)+(1/dt)*ppre1))
        #phi2=spsolve(Mk2,M.dot(gamma*ppre1+(1/dt)*ppre2))
        
        p1[:,k]=phi1
        #p2[:,k]=phi2
        
        ppre1=phi1
        #ppre2=phi2
        
        #gradient for coefficients
        g[:,:,k]=db(ytk,alpha[:,:,k],z,ppre1)[1]+2*ga*db(ytk,alpha[:,:,k],z,b(ytk,alpha[:,:,k],z))[1]
        
        #gradient for nodes 
        gz=gz+dt*db(ytk,alpha[:,:,k],z,ppre1)[2]+dt*2*ga*db(ytk,alpha[:,:,k],z,b(ytk,alpha[:,:,k],z))[2]
        
        
    c.append(p1)
    c.append(p2)
    c.append(g)
    c.append(gz)
    
    return c
    
    