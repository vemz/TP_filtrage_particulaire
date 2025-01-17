import numpy as np
import matplotlib.pyplot as plt

#creer l'observation
x0=0
Q=10
T=50
N=50

def f(x_prec,t):
    return 0.5*x_prec+25*x_prec/(1+x_prec**2)+8*np.cos(1.2*(t))


def creation_trajectoire(x0,Q,T):
    x=np.zeros(T)
    x[0]=x0
    for i in (1,T):
        x[i]=f(x[i-1],i)+np.random.normal(0,Q)
    return x

#trajectoire
x=creation_trajectoire(x0,Q,T)

#creer observation
R=1

def g(x):
    return x**2/20

def creation_observation(x,R):
    y=np.zeros(T)
    for i in range(T):
        y[i]=g(x[i])+np.random.normal(0,R)
    return y

#observation
y=creation_observation(x,R)

#filtrage
def filtrage_particulaire_m(x_part,W_part,y,R,N,T):
    x_filtre=np.zeros(T)
    W_filtre=np.zeros((T,N))
    x_filtre[0]=np.zeros(T)
    W_filtre[0]=np.ones(N)/N
    for i in range(1,T):
        for i in range (N):
            x_part[i]=f(x_part[i-1],i)+np.random.normal(0,Q,N)
