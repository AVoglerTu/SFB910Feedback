'''
Created on 04.09.2018

@author: Alex
'''

from adjCoupled import adjoint
from stateCoupled import solve
from coefficientsANDcost import b, cost
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle

def opt(T,a,N,y0,z0,alpha0,yref,yT,dx,dt,bd,delta,gamma,eta,Dy,Dz,eps,det,M,eps2,z,Jc):
    
    nt=int(T/float(dt))
    nx=int(a/float(dx))
    
    #reset counter... reset if to many failed attempts on a batch of samples
    res=0
    
    #success counter to increase stepsize if step size is accepted at least 10 times in a row
    successcount=0
    
    #set compute gradient to true
    cg=1
    
    #plot counter
    pcount=0
    
    #plot index counter
    pindex=0
    
    
    #set N=1 if the deterministic case is considered
    if det==1:
        N=1
    
    
    print('calculating initial state')
    w=eps*np.sqrt(4/float(6))*np.sqrt(dt)*np.sqrt(dx)*(np.random.normal(0,1,[nx,nt]))
    #calculate state and adjoint for a given sample
    y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,alpha0,Dy,Dz,z)
    
    #uncontrolled solution
    yold=y
    
    #default gradient norm
    gnorm=1
    
    #initial step size
    s=0.01
    
    #current control
    acur=alpha0
    
    #control before gradient si applied
    alphOld=alpha0
    
    zcur=z
    zold=z
    
    #create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                    
    #create first subplot
    ax = fig.add_subplot(1, 3, 1, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = y
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    
    plt.savefig('initial state' +'.png')
    plt.close()
    
    #list for old noise relizations
    wold=[]
    
    #gradients
    g=np.zeros([M,nx,nt])
    gz=np.zeros([M])
    
    print('start gradient decent')
    
    #iterate until norm of gradient is smaller than threshold
    while gnorm>eps2:
        
        diff=-1
        
        print('current gradient norm:')
        print(gnorm)
        
        #update with given step size if cost functional is smaller
        while diff<0:
            
            #if cost is still higher after step size decreased 20 times, reset and calculate gradient for new sample
            if res>20:
                s=0.01
                cg=1
                res=0
            
            #if step size was accepted more than 10 times in a row, increase step size
            if successcount>20:
                s=2*s
                cg=1
                successcount=0
                
            
            print('computing gradient')
            
            #if compute gradient=true -> compute gradient for new samples, otherwise skip and compute cost for lower step size
            if cg==1:
                
                #cost
                J=0
                
                #list for old noise relizations
                wold=[]
                
                #gradients
                g=np.zeros([M,nx,nt])
                gz=np.zeros([M])
                
                #calculate gradient at current control (acur,zcur)
                for k in range(N):
                    
                    print('process:' + str((k+1)/float(N)*100) + '%')
                    
                    #samples of Q-Wiener process for Monte-Carlo
                    if det==0:
                        w=eps*np.sqrt(4/float(6))*np.sqrt(dt)*np.sqrt(dx)*(np.random.normal(0,1,[nx,nt]))
                        wold.append(w) 
                    else:
                        w=np.zeros([nx,nt])
                        wold.append(w) 
                        
                    #calculate state and adjoint for a given sample
                    y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,acur,Dy,Dz,zcur)
                
                    
                    #cost functional
                    J=J+cost(y,acur,yref,yT,zcur)
                    
                    p=adjoint(M,T,a,y,yref,yT,dx,dt,bd,acur,delta,gamma,eta,Dy,Dz,zcur)
                    
                    #approximation gradient
                    g=g+p[2]
                    gz=gz+p[3]
                
                #sample of solution before altering control
                yold=y
                
                #calculate cost
                Jcur=(1/N)*J
    
                print('base cost:')
                print(Jcur)

                
                #calculate decent direction
                g=(1/N)*g
                gz=(1/N)*gz
                
                gn=dx*dt*LA.norm(g)+LA.norm(gz)
                
                d1=-g
                d2=-gz
            
            #calculate new control w.r.t. step current step size
            anew=acur+s*d1
            znew=zcur+s*d2
            
            #calculate cost functional with new control
            Jnew=0
            
            print('calculating new cost')
            
            for k in range(N):
                
                print('process:' + str((k+1)/float(N)*100) + '%')
                
                #samples of Q-Wiener process for Monte-Carlo
                if det==0:
                    w=wold[k]
                else:
                    w=np.zeros([nx,nt])
                    
                #calculate state and adjoint for a given sample and updated control
                y=solve(T,a,y0,z0,dx,dt,w,bd,delta,gamma,eta,anew,Dy,Dz,znew)

                #cost functional
                Jnew=Jnew+cost(y,anew,yref,yT,znew)
        
            #calcvulate cost via Monte-Carlo
            Jnew=(1/N)*Jnew
            
            print('new cost')
            print(Jnew)
            
            #if cost functional is smaller, accept new control, else new step size
            if Jnew<=Jcur:
                
                successcount=successcount+1
                
                res=0
                
                pcount=pcount+1
                
                #compute new gradient next step
                cg=1
                
                print('new control accepted')
                
                #append new cost
                Jc.append(Jcur)
                                
                #calculate norm of gradient
                gnorm=gn
                
                #update control and cost functional
                alphOld=acur
                zold=zcur
                acur=anew
                zcur=znew
                Jcur=Jnew
                diff=Jcur-Jnew
                
                if pcount>10:
                    
                    #increase plot index
                    pindex=pindex+1
                    
                    #control plot
                    uc=np.zeros([nx,nt])
                    ucold=np.zeros([nx,nt])
                    
                    for r in range(nt):
                        uc[:,r]=b(y[:,r],acur[:,:,r],zcur)
                        ucold[:,r]=b(y[:,r],alphOld[:,:,r],zold)
                    
                    
                    
                    #create figure
                    fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                    
                    #create first subplot
                    ax = fig.add_subplot(2, 3, 1, projection='3d')
                        
                    X = np.arange(0, T,dt)
                    Y = np.arange(0, a,dx)
                    X, Y = np.meshgrid(X, Y)
                    Z = y
                    
                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                    plt.ylabel('x')
                    plt.xlabel('t')
                    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    #create first subplot
                    ax = fig.add_subplot(2, 3, 2, projection='3d')
                        
                    X = np.arange(0, T,dt)
                    Y = np.arange(0, a,dx)
                    X, Y = np.meshgrid(X, Y)
                    Z = yold
                    
                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                    plt.ylabel('x')
                    plt.xlabel('t')
                    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    
                    #create second subplot
                    ax = fig.add_subplot(2, 3, 3, projection='3d')
                        
                    X = np.arange(0, T,dt)
                    Y = np.arange(0, a,dx)
                    X, Y = np.meshgrid(X, Y)
                    Z = p[0]
                    
                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                    plt.ylabel('x')
                    plt.xlabel('t')
                    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    #create second subplot
                    ax = fig.add_subplot(2, 3, 4, projection='3d')
                        
                    X = np.arange(0, T,dt)
                    Y = np.arange(0, a,dx)
                    X, Y = np.meshgrid(X, Y)
                    Z = ucold
                    
                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                    plt.ylabel('x')
                    plt.xlabel('t')
                    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    #create second subplot
                    ax = fig.add_subplot(2, 3, 5, projection='3d')
                        
                    X = np.arange(0, T,dt)
                    Y = np.arange(0, a,dx)
                    X, Y = np.meshgrid(X, Y)
                    Z = uc
                    
                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
                    plt.ylabel('x')
                    plt.xlabel('t')
                    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    
                    
                    plt.savefig('state, adjoint and control(altered)'+ 'k=' + str(pindex) +'.png')
                    plt.close()
                    
                    
                    plt.plot(zcur)
                    plt.savefig('nodes' + str(pindex) + '.png')
                    plt.close()
                    
                    plt.plot(Jc)
                    plt.savefig('cost'+ '.png')
                    plt.close()
                    
                    #save feedback function
                    with open("feedback.pkl","wb") as f:
                        pickle.dump(acur, f, pickle.HIGHEST_PROTOCOL)
                    
                    with open("nodes.pkl","wb") as f:
                        pickle.dump(zcur, f, pickle.HIGHEST_PROTOCOL)
                        
                    with open("cost.pkl","wb") as f:
                        pickle.dump(Jc, f, pickle.HIGHEST_PROTOCOL)
                    
                    pcount=0
            
            else:
                
                #increase reset counter
                res=res+1
                
                #decrease step size
                print('decrease step-size')
                s=(1/2)*s
                print('new step size:')
                print(s)
                
                #compute gradient=false -> calculate new cost with old gradient in next step
                cg=0
               

            
    print(gnorm)         
    
    #solution with optimal control
    #increase index
    pindex=pindex+1
                    
    #control plot
    uc=np.zeros([nx,nt])
                    
    for r in range(nt):
        uc[:,r]=b(y[:,r],acur[:,:,r],zcur)
                    
                    
                    
    #create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
                    
                    
    #create first subplot
    ax = fig.add_subplot(1, 3, 1, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = y
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
                    
    #create second subplot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = p[0]
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
    #create second subplot
    ax = fig.add_subplot(1, 3, 3, projection='3d')
                        
    X = np.arange(0, T,dt)
    Y = np.arange(0, a,dx)
    X, Y = np.meshgrid(X, Y)
    Z = uc
                    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.ylabel('x')
    plt.xlabel('t')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
                    
    plt.savefig('state, adjoint and control'+ 'k=' + str(pindex) +'.png')
    plt.close()
                    
    for k in range(M):
        plt.plot(zcur[k,:])
    plt.savefig('nodes' + str(pindex) + '.png')
                    
    #save feedback function
    with open("feedback.pkl","wb") as f:
        pickle.dump(acur, f, pickle.HIGHEST_PROTOCOL)
                    
    with open("nodes.pkl","wb") as f:
        pickle.dump(zcur, f, pickle.HIGHEST_PROTOCOL)
