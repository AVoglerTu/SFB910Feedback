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
import pickle
from GalerkinSetting import laplace
from solveRicc import solveStateRiccRef, solveState0
from solveRicc import Ricc, Ricc2
from generateRef import ref


#gplot=[]
#cost=[]

#initial condition
u0=np.sqrt(L/float(K))*scipy.fftpack.dct(uinit(a),type=2, norm='ortho')

yref=np.zeros([nt,K])


for r in range(nt):
    yref[r,:]=u0

#reference profiles
yT=yref[nt-1,:]

V=Ricc()

phi=Ricc2(V,yref)

w=np.zeros([nt,K])

w=(tf.random.normal([nt,K],0,dt*eps,dtype=tf.float32))

sol=solveStateRiccRef(u0,w,V,phi,yref,yT)
sol0=solveState0(u0, w, yref, yT)

c=0
for r in range(nt):
    
    c=c+dt*(1/float(2))*LA.norm(sol[0][r,:])**2+dt*(1/float(2))*tf.norm(sol[1][r,:])**2

c=c+(1/float(2))*LA.norm(sol[0][nt-1,:])**2

print('opt cost')
print(c)

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
#for i in range(200):
    
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

u=np.zeros([nt,nx])
sol1=np.zeros([nt,nx])
sol2=np.zeros([nt,nx])

for r in range(nt):

    u[r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol[1][r,:],type=2,norm='ortho')
    sol1[r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol[0][r,:],type=2,norm='ortho')
    sol2[r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol0[r,:],type=2,norm='ortho')
    
#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                        
#create first subplot
ax = fig.add_subplot(1, 3, 1, projection='3d')
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = u
                
#levels = np.linspace(-0.1, 0.1, 7)
                
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    

#create first subplot
ax = fig.add_subplot(1, 3, 2, projection='3d')
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = sol1
                
#levels = np.linspace(-0.1, 0.1, 7)
                
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    


#create first subplot
ax = fig.add_subplot(1, 3, 3, projection='3d')
                
                            
X = np.arange(0, T,dt)
Y = a
Y, X = np.meshgrid(Y, X)
Z = sol2
                
#levels = np.linspace(-0.1, 0.1, 7)
                
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
plt.savefig('controlState' + '.png')
plt.close()

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

