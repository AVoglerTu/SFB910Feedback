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
Nsim=1

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


#for k in range(Nsim):
    
 #   #noise realization
#    w=(np.random.normal(0,eps*4/float(6)*(dx/float(dt)),[nx,nt])) 
#    #w=k*(1/float(25))*np.ones([nx,nt])
#    #uncontrolled state
#    y1=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)
#    xu.append(y1)
#    
#   #solve equation w.r.t. feedback
#    y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,alpha,Dy,Dz,z)
#    x.append(y)

#    #control 
#    uc=np.zeros([nx,nt])
                    
#    for r in range(nt):
#        uc[:,r]=b(y[:,r],alpha[:,:,r],z)
#        
#    u.append(uc)
  
for k in range(Nsim):
    
    l=np.random.randint(0,M)
    
    #control 
    uc=np.zeros([nx,nt])
    x.append(z[l,:])                
    for r in range(nt):
        uc[:,r]=b(z[l,:],alpha[:,:,r],z)
        
    u.append(uc)  

for k in range(Nsim):
    
    #create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                        
    plt.plot(x[k])
                    
    
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
                    