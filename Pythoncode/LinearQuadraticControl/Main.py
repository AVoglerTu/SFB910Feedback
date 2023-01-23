'''
Created on 08.11.2022

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
from solveRicc import Ricc, Ricc2, solveStateRiccRef

#non sequential model
#inputs=tf.keras.layers.Input(shape=(K+1))
#l1=tf.keras.layers.Dense(100, activation='ReLU')(inputs)
#l2=tf.keras.layers.Dense(100, activation='ReLU')(l1)
#output layer
#out=tf.keras.layers.Dense(K, activation='linear')(l2)

#model=tf.keras.Model(inputs=inputs, outputs = out)


#model = tf.keras.models.Sequential() # initialize and construct neural network
#model.add(tf.keras.layers.Input(shape=(K+1)))
#model.add(tf.keras.layers.Dense(100, activation='ReLU'))
#model.add(tf.keras.layers.Dense(100, activation='ReLU'))
#output layer
#model.add(tf.keras.layers.Dense(K, activation='linear'))

#load old model
model=tf.keras.models.load_model('C:/Users/alexa/eclipse-workspace/LQControlSPDE/modelLQ50')

model.summary()
#print(model.layers[0].get_weights())
#print(model.layers[1].get_weights())
#print(model.layers[2].get_weights())



#load old gradient
with open('grad.pickle', 'rb') as g:
    gplot=pickle.load(g)

#load old gradient
with open('L2.pickle', 'rb') as g:
    L2diff=pickle.load(g)

#load old cost
with open('cost.pickle', 'rb') as c:
    cost=pickle.load(c)

with open('normalization.pickle', 'rb') as g:
    normalize=pickle.load(g)

solMin=normalize[0]
solMax=normalize[1]
normal=normalize[2]

#gplot=[]
#cost=[]
#L2diff=[]

#initial condition
u0=np.sqrt(L/float(K))*scipy.fftpack.dct(uinit(a),type=2, norm='ortho')

#reference profiles
yref=np.zeros([nt,K])
yT=yref[nt-1,:]


V=Ricc()

phi=Ricc2(V,yref)



#plref=np.zeros([nt,nx])
#for r in range(nt):
#    plref[r,:]=np.sqrt(K/float(L))*tf.signal.idct(yref[r,:],type=2,norm='ortho')

#create figure
#fig = plt.figure(figsize=plt.figaspect(0.5))

#create third subplot
#ax = fig.add_subplot(1,1,1, projection='3d')
                            
#X = np.arange(0, T,dt)
#Y = a
#Y, X = np.meshgrid(Y, X)
#Z = plref
                        
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
#plt.ylabel('x')
#plt.xlabel('t')
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
#plt.savefig('ref'+ '.png')
#plt.close()


#sample for normalization
#for i in range(20):
    
#    solMax=np.zeros([K])
#    solMin=20*np.ones([K])
    
    #sample signal data
#    w=(tf.random.normal([nt,K],0,dt*eps,dtype=tf.float32))
        
#    sol=solveStateUc(u0,w)
    
#    for k in range(K):
        
#        if solMin[k] > np.amin(sol[:,k]):
#            solMin[k]=np.amin(sol[:,k])
            
#        if solMax[k]< np.amax(sol[:,k]):
#            solMax[k]=np.amax(sol[:,k])


#normal=1./(solMax-solMin)

#normalization=[]
#normalization.append(solMin)
#normalization.append(solMax)
#normalization.append(normal)

#save data scaling constants
#with open('normalization.pickle', 'wb') as g:
#    pickle.dump(normalization,g, protocol=pickle.HIGHEST_PROTOCOL)


#solMin=0
#solMax=1

#normal=1


print(normal)
print(solMin)
print(solMax)


opt(u0,yref,yT,model,gplot,cost,L2diff,solMin,solMax,normal,V,phi)

#w=np.zeros([nt,K])

#sol=solveState(u0,model,w)

#p=adjoint(sol,model,yT,yref)

#g=grad(p,sol,model)

#y=np.zeros([nt,nx])
#z=np.zeros([nt,nx])

#for r in range(nt):
#    y[r,:]=scipy.fftpack.idct(sol[r,:],type=2, norm='ortho')
#    z[r,:]=scipy.fftpack.idct(p[r,:],type=2, norm='ortho')

#create figure
#fig = plt.figure(figsize=plt.figaspect(0.5))
                    
#create first subplot
#ax = fig.add_subplot(1, 3, 1, projection='3d')
                        
#X = np.arange(0, T,dt)
#Y = a
#Y, X = np.meshgrid(Y, X)
#Z = y
                    
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
#plt.ylabel('x')
#plt.xlabel('t')
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)

#create second subplot
#ax = fig.add_subplot(1, 3, 2, projection='3d')
                        
#X = np.arange(0, T,dt)
#Y = a
#Y, X = np.meshgrid(Y, X)
#Z = z
                    
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
#plt.ylabel('x')
#plt.xlabel('t')
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    

#create third subplot
#ax = fig.add_subplot(1, 3, 3, projection='3d')
                        
#X = np.arange(0, T,dt)
#Y = a
#Y, X = np.meshgrid(Y, X)
#Z = y-0.1*z
                    
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
#plt.ylabel('x')
#plt.xlabel('t')
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)                  
  
#plt.savefig('sol' +'.png')
#plt.close()

