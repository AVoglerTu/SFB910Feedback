'''
Created on 06.07.2022

@author: alexa
'''
from coefficientsANDcost import *
from stateCoupled import solve

w=(np.random.normal(0,eps*4/float(6)*(dx/float(dt)),[nx,nt]))


#control coefficients
alpha2=0*np.ones([M,nx,nt])

#cost vector
#Jc=[]


#number of basis functions for radial basis approximation
M=20

#control nodes
z=np.zeros([M,nx])


y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,alpha2,Dy,Dz,z)


#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                    
#create first subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
                        
X = np.arange(0, T,dt)
Y = np.arange(0, a,dx)
X, Y = np.meshgrid(X, Y)
Z = y
                    
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('t')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    
plt.savefig('sol' +'.png')
plt.close()

