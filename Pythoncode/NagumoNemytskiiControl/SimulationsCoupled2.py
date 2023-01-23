'''
Created on 07.09.2018

@author: Alex
'''

import pickle
from gradientDecent import opt
from coefficientsANDcost import *


#load old control
with open('feedback.pkl', 'rb') as f:
    alpha2 = pickle.load(f)


#control coefficients
#alpha2=0*np.ones([M,nx,nt])



#number of basis functions for radial basis approximation
M=20

#control nodes
z=np.zeros([M,nx])

#initialize radial basis control 'nonlinear part of control'
for k in range(M):
    #w=(np.random.normal(0,eps*(dx/dt)*(4/6),[nx,nt]))
    #w=np.zeros([nx,nt])
    #if k<int(M/4):
    #    z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(nt/4)]
    #if int(M/4)<k<int(M/2):
    #    z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(nt/2)]
    #if int(M/2)<k<int(3*M/4):
    #    z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(3*nt/4)]
    #if k>int(3*M/4):
    #    z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(9*nt/10)]
    z[k,:]=(1/(1+k)+0.5)*np.ones([nx])
for k in range(M):
    plt.plot(z[k,:])
plt.show()


#save feedback function
with open("radial.pkl","wb") as f:
    pickle.dump(z, f, pickle.HIGHEST_PROTOCOL)

with open('nodes.pkl', 'rb') as f:
    z = pickle.load(f)



opt(T,a,N,y0,z0,alpha2,yref,yT,dx,dt,bd,delta,gamma,eta,Dy,Dz,eps,det,M,eps2,z)