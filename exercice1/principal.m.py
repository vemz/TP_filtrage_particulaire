import numpy as np

def f(x_prec, n):
    return 0.5*x_prec + 25*x_prec/(1 + x_prec**2) + 8*np.cos(1.2*n)

def creation_trajectoire(x0, Q, T):
    x = np.zeros(T)
    x[0] = x0
    for i in range(1, T):
        x[i] = f(x[i-1], i) + np.random.normal(0, Q)
    return x

# Initialisation des paramètres
x0 = 0 
Q = 1   
T = 50  

# Création de la trajectoire
x = creation_trajectoire(x0, Q, T)

# Créer observation
R = 1

def g(x):
    return x**2 / 20

def creation_observation(x,R):
    y=np.zeros(T)
    for i in range(T):
        y[i]=g(x[i])+np.random.normal(0,R)
    return y

#observation
y=creation_observation(x,R)

def multinomial_resample(weights):

    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))

#filtrage
def filtrage_particulaire_m(x_part, W_part, y, R, N, T, n):
    x_filtre = np.zeros(N)
    W_filtre = np.zeros(N)
    for i in range(N):
        x_filtre[i] = f(x_part[i], n) + np.random.normal(0, Q)
        W_filtre[i] = 1/np.sqrt(2*np.pi*R)*np.exp(-0.5*(y[i]-g(x_filtre[i]))**2/R)
    for i in range(N):
        W_filtre[i] = W_filtre[i]/np.sum(W_filtre)
    W_filtre = multinomial_resample(W_filtre)
    x_estime = np.sum(W_filtre * x_filtre)
    return x_estime, x_filtre, W_filtre


            
            
            
            
            

