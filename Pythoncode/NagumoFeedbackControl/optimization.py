'''
Created on 08.11.2022

@author: alexa
'''

import pickle 
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
from SolveSpde import solveStateUc
from SolveAdjoint import grad
import sys
np.set_printoptions(threshold=sys.maxsize)


#construct adaptive stochastic gradient descent optimizer
optimizer = tf.keras.optimizers.SGD(lr=0.0000001)



def opt(u0,yref,yT,model,gplot,cost,solMin,solMax,normal):
    
    count=0
    
    #default gradient norm
    gnorm=1
    
    while gnorm>gamma:
        
        pltcount=0
        
        #default gradient norm
        gnorm=1
        
        gradPlot=gplot
        costPlot=cost
    
        while gnorm>gamma:
            
            print(pltcount)
            
            pltcount=pltcount+1
            
            #sample signal data
            w=(tf.random.normal([nt,K],0,dt*eps,dtype=tf.float32))
        
            sol=solveState(u0,model,w,solMin,solMax,normal)
            p=adjoint(sol,model,yT,yref,solMin,solMax,normal)
            g=grad(p,sol,model,yT,yref,solMin,solMax,normal)
            gr=g[0]
            gradPlot.append(LA.norm(g[1][0]))
            costPlot.append(g[2])
            # optimizer applies update
            optimizer.apply_gradients(zip(gr,model.trainable_variables))
        
            count=count+1
            
            if pltcount>10:
                
                sol=solveState(u0,model,w,solMin,solMax,normal)
                soluc=solveStateUc(u0,w)
                
                #evaluated feedback control
                u=np.zeros([nt,nx])
                
                #evaluated solution
                sol1=np.zeros([nt,nx])
                
                #evaluated adjoint
                adj1=np.zeros([nt,nx])
                
                #uncontrolled solution
                sol2=np.zeros([nt,nx])
                
                pltcount=0
                
                plt.plot(gradPlot)
                plt.title('current grad='+ str(g[1][0]))
                plt.xlabel('iterations')
                plt.savefig('gradient')
                plt.close()
            
                plt.plot(costPlot)
                plt.title('current cost=' + str(g[2]))
                plt.xlabel('iterations')
                plt.ylim(0,30)
                plt.savefig('cost')
                plt.close()
                
                for r in range(1,nt):
                    r1=tf.cast(ti[r], tf.double) 
                    x1=tf.Variable([scale2*np.multiply(sol[r,:]-solMin,normal)])
                    r1=tf.Variable([[scale*r1]])
                 
                    control=model(tf.concat([r1,x1], axis=1))
                    
                    u[r,:]=np.sqrt(K/float(L))*tf.signal.idct(control,type=2,norm='ortho')
                    sol1[r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol[r,:],type=2,norm='ortho')
                    sol2[r,:]=np.sqrt(K/float(L))*tf.signal.idct(soluc[r,:],type=2,norm='ortho')
                    adj1[r,:]=np.sqrt(K/float(L))*tf.signal.idct(p[r,:],type=2,norm='ortho')
                
                print(np.amax(u))
                
                #create figure
                fig = plt.figure(figsize=plt.figaspect(0.5))
                        
                #create first subplot
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                            
                X = np.arange(0, T,dt)
                Y = a
                Y, X = np.meshgrid(Y, X)
                Z = u
                
                #levels = np.linspace(-0.1, 0.1, 7)
                
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                plt.ylabel('x')
                plt.xlabel('t')
                fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
                plt.savefig('control3D' + str(count)+ '.png')
                plt.close()
                
                #create figure
                fig = plt.figure(figsize=plt.figaspect(0.5))
                        
                #create first subplot
                ax = fig.add_subplot(1, 1, 1)
                
                            
                X = np.arange(0, T,dt)
                Y = a
                Y, X = np.meshgrid(Y, X)
                Z = u
                
                #levels = np.linspace(-0.1, 0.1, 7)
                
                cp = ax.contourf(X, Y, Z)
                fig.colorbar(cp) 
                plt.ylabel('x')
                plt.xlabel('t')
                
                plt.savefig('control' + str(count)+ '.png')
                plt.close()
                
                #create figure
                fig = plt.figure(figsize=plt.figaspect(0.5))
                
                #create first subplot
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                            
                X = np.arange(0, T,dt)
                Y = a
                Y, X = np.meshgrid(Y, X)
                Z = sol2
                        
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                plt.ylabel('x')
                plt.xlabel('t')
                fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
                #create third subplot
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                            
                X = np.arange(0, T,dt)
                Y = a
                Y, X = np.meshgrid(Y, X)
                Z = sol1
                        
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                plt.ylabel('x')
                plt.xlabel('t')
                fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
                plt.savefig('controlledSol' + str(count)+ '.png')
                plt.close()
                
                #create figure
                fig = plt.figure(figsize=plt.figaspect(0.5))
                
                #create first subplot
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                            
                X = np.arange(0, T,dt)
                Y = a
                Y, X = np.meshgrid(Y, X)
                Z = adj1
                        
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                plt.ylabel('x')
                plt.xlabel('t')
                fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
                plt.savefig('adjoint' + str(count)+ '.png')
                plt.close()
                
                with open('grad.pickle', 'wb') as g:
                    pickle.dump(gradPlot,g, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open('cost.pickle', 'wb') as g:
                    pickle.dump(costPlot,g, protocol=pickle.HIGHEST_PROTOCOL)  
                
                #save model
                model.save('C:/Users/alexa/eclipse-workspace/SpectralControl/model44')
