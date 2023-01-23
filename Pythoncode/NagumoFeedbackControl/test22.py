'''
Created on 29.11.2022

@author: alexa
'''
import tensorflow as tf
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
from CoefficientsAndCost import *
from generateRef import ref
from SolveAdjoint import adjoint
from SolveSpde import solveState
from SolveAdjoint import grad
from optimization import opt
from SolveSpde import solveStateUc
import pickle



with open('normalization.pickle', 'rb') as g:
    normalize=pickle.load(g)

solMin=normalize[0]
solMax=normalize[1]
normal=normalize[2]

#load old model
model=tf.keras.models.load_model('C:/Users/alexa/eclipse-workspace/SpectralControl/model18')

#initial condition
u0=np.sqrt(L/float(K))*scipy.fftpack.dct(uinit(a),type=2, norm='ortho')

#reference profiles
yref=ref(u0)
yT=yref[nt-1,:]

#evaluated feedback control
u=np.zeros([nt,nx])
#evaluated solution
sol1=np.zeros([nt,nx])

w=np.zeros([nt,K])
sol=solveState(u0,model,w,solMin,solMax,normal)

for r in range(1,nt):
    r1=tf.cast(ti[r], tf.double) 
    x1=tf.Variable([scale2*np.multiply(yref[r,:]-solMin,normal)])
    r1=tf.Variable([[scale*r1]])
                 
    control=model(tf.concat([r1,x1], axis=1))
                    
    u[r,:]=np.sqrt(K/float(L))*tf.signal.idct(control,type=2,norm='ortho')
    sol1[r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol[r,:],type=2,norm='ortho')

#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                        
#create first subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = u
                
                
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
plt.savefig('controlTse3D'+ '.png')
plt.close()

#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))

#create first subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = sol1
                
                
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
plt.savefig('solTse3D'+ '.png')
plt.close()
                
#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                        
#create first subplot
ax = fig.add_subplot(1, 1, 1)
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = u
                
                
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) 
plt.ylabel('x')
plt.xlabel('t')
                
plt.savefig('controlTest' + '.png')
plt.close()


#def g(x):
#    return -x**3+x

#a=np.arange(0,1,0.001)

#K=a.size

#plt.plot(np.sqrt(K)*scipy.fftpack.idct(np.sqrt(1/float(K))*scipy.fftpack.dct(g(a),type=2, norm='ortho'),type=2,norm='ortho'))
#plt.show()