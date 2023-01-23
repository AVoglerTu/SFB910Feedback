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
    
    
v=np.arange(0,1.2,0.005)
L=v.size

y=np.zeros([nx,L])

uc=np.zeros([nx,L])

for k in range(nx):
    
    y[k,:]=v
    
for r in range(L):
    #control 
    uc[:,r]=b(y[:,r],alpha[:,:,1600],z)



#create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
                        
#create first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

X = np.arange(0, 1.2,0.005)
Y = np.arange(0, a,dx)                         
X, Y = np.meshgrid(X, Y)
Z = uc
                        
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
plt.ylabel('x')
plt.xlabel('u')
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)


plt.savefig('feedbackfck' + '.png')
plt.close()

