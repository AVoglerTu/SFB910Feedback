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
from scipy import integrate
from scipy import fftpack
from CoefficientsAndCost import *

#define laplacian derivative matrix depending on boundary condition
def laplace():
    a=np.identity(K)
    
    #Dirichlet boundary conditions
    if bd==0:
        l=np.arange(1,K+1,1)
        a=np.pi**2*np.multiply(l**2,a/float(L)**2)
    
    #Neumann boundary conditions  
    if bd==1:
        l=np.arange(0,K,1)
        a=np.pi**2*np.multiply(l**2,a/float(L)**2)

    #periodic boundary conditions
    if bd==2:
        for l in range(K):
            if isOdd(l+1):
                a[l,:]=a[l,:]*np.pi**2*l**2/float(L)**2
            else:
                a[l,:]=a[l,:]*np.pi**2*(l+1)**2/float(L)**2
    
    return a

     