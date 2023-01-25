'''
Created on 14.04.2022

@author: alexa
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from coefficientsANDcost import *
from stateCoupled import solve

#number of simulations
Nsim=6

#load control
with open('feedback.pkl', 'rb') as f:
    alpha = pickle.load(f)


with open('nodes.pkl', 'rb') as f:
    z = pickle.load(f)
    
#list for control realizations
u=[]

#list for state realizations
x=[]

#list for uncontrolled states
xu=[]


for k in range(Nsim):
    
    #noise realization
    w=eps*np.sqrt(4/float(6))*np.sqrt(dt)*np.sqrt(dx)*(np.random.normal(0,1,[nx,nt]))
    #w=np.zeros([nx,nt])
    #uncontrolled state
    y1=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)
    xu.append(y1)
    
    #solve equation w.r.t. feedback
    y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,alpha,Dy,Dz,z)
    x.append(y)

    #control 
    uc=np.zeros([nx,nt])
                    
    for r in range(nt):
        uc[:,r]=b(y[:,r],alpha[:,:,r],z)
        
    u.append(uc)
    

for k in range(Nsim):
    
    #create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                    
    #create first subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = xu[k]
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
    
    
    #create second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = x[k]
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
    
    plt.savefig('uncontrolled and controlled state'+ 'k=' + str(k) +'.png')
    plt.close()
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    #create third subplot
    ax = fig.add_subplot(1, 1, 1)
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = u[k]
                
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) 
    plt.ylabel('x')
    plt.xlabel('t')
                    
    plt.savefig('control'+ 'k=' + str(k) +'.png')
    plt.close()
                    
