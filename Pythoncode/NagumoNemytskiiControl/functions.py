'''
Created on 25.08.2018

@author: Alex
'''
import numpy as np
from scipy.sparse import diags
from scipy import sparse

def matM(nx,h,bd):
    
    if bd==0:
    
        M=np.zeros([nx,nx])
    
        for i in range(nx):
            for j in range(nx):
            
                if i==j:
                
                    M[i][j]=4*h/6
                
                if i==j+1 or i==j-1:
                
                    M[i][j]=h/6
                
        M[0,:]=0
        M[nx-1,:]=0
        
    else:
        
        M=np.zeros([nx,nx])
    
        for i in range(nx):
            for j in range(nx):
            
                if i==j:
                
                    M[i][j]=4*h/6
                
                if i==j+1 or i==j-1:
                
                    M[i][j]=h/6
        
        M[0,0]=2*h/6
        M[0,1]=h/6
        M[nx-1,nx-1]=2*h/6
        M[nx-1,nx-2]=h/6
        
    Ms=sparse.csr_matrix(M)  
                  
    return Ms


def matK(nx,h,bd):
    
    if bd==0:
    
        K=np.zeros([nx,nx])
    
        for i in range(nx):
            for j in range(nx):
            
                if i==j:
                
                    K[i][j]=2/h
                
                if i==j+1 or i==j-1:
                
                    K[i][j]=-1/h
    
    
        K[0,0]=1/h
        K[0,1]=0
        K[nx-1,nx-1]=1/h
        K[nx-1,nx-2]=0
    
    else:
        
        K=np.zeros([nx,nx])
    
        for i in range(nx):
            for j in range(nx):
            
                if i==j:
                
                    K[i][j]=2/h
                
                if i==j+1 or i==j-1:
                
                    K[i][j]=-1/h
    
    
        K[0,0]=1/h
        K[0,1]=-1/h
        K[nx-1,nx-1]=1/h
        K[nx-1,nx-2]=-1/h
                
    Ks=sparse.csr_matrix(K)            
    return Ks

#v vector for diagonal elements with nx length
          
def sparseDiag(v):
    
    return diags(v, 0)
    
    
print(matM(5,1,0))
print(matK(5,1,0))

