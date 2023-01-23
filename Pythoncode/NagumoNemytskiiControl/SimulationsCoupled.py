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

with open('cost.pkl', 'rb') as f:
    Jc=pickle.load(f)


#control coefficients
#alpha2=np.random.normal(0,0.0001,[M,nx,nt])
#alpha2=0*np.ones([M,nx,nt])

#cost vector
#Jc=[]



#control nodes
#z=np.zeros([M])

#for k in range(M):
#    z[k]=k*0.03

#initialize radial basis control 'nonlinear part of control'
#for k in range(M):
#    #w=(np.random.normal(0,eps*(dx/dt)*(4/6),[nx,nt]))
#    w=np.zeros([nx,nt])
#    if k<int(M/4):
#        z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(nt/4)]
#    if int(M/4)<k<int(M/2):
#        z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(7*nt/10)]
#    if int(M/2)<k<int(3*M/4):
#        z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(3*nt/4)]
#    if k>int(3*M/4):
#        z[k,:]=solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,int(9*nt/10)]
    #z[k,:]=(1/(1+k)+0.5)*np.ones([nx])




#for k in range(M-2):
#    w=np.zeros([nx,nt])
#    r=np.random.randint(0,nt)
#    z[k,:]=(r/float(nt))*solveUc(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,Dy,Dz,0)[:,r]
    
#z[M-2,:]=np.ones([nx])
#z[M-1,:]=np.zeros([nx])

#for k in range(M):
#    plt.plot(z[k,:])
#plt.show()

#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                    
#create first subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
                        
X = np.arange(0, T,dt)
Y = np.arange(0, a,dx)
X, Y = np.meshgrid(X, Y)
Z = yref
                    
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)

plt.savefig('runningReference' + '.png')
plt.close()

plt.plot(yT)
plt.savefig('terminalReference' + '.png')
plt.close()

#save feedback function
#with open("radial.pkl","wb") as f:
#    pickle.dump(z, f, pickle.HIGHEST_PROTOCOL)

with open('nodes.pkl', 'rb') as f:
    z = pickle.load(f)



opt(T,a,N,y0,z0,alpha2,yref,yT,dx,dt,bd,delta,gamma,eta,Dy,Dz,eps,det,M,eps2,z,Jc)