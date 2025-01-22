import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

def f(x_prec, n):
    return 0.5*x_prec + 25*x_prec/(1 + x_prec**2) + 8*np.cos(1.2*n)

def creation_trajectoire(x0, Q, T):
    x = np.zeros(T)
    x[0] = x0
    for i in range(1, T):
        x[i] = f(x[i-1], i) + np.random.normal(0, Q)
    return x

x0 = 0 
Q = 10
T = 50  
N = 50


x = creation_trajectoire(x0, Q, T)


R = 1

def g(x):
    return x**2 / 20

def creation_observation(x, R):
    y = np.zeros(T)
    for i in range(T):
        y[i] = g(x[i]) + np.random.normal(0, R)
    return y

# Observation
y = creation_observation(x, R)

def multinomial_resample(weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.
    return np.searchsorted(cumulative_sum, random(len(weights)))

def filtrage_particulaire_m(x_part, y, W_part, R, N, Q, T, n):
    x_filtre = np.zeros(N)
    W_filtre = np.zeros(N)
    for i in range(N):
        x_filtre[i] = f(x_part[i], n) + np.random.normal(0, Q)
        W_filtre[i] = 1/np.sqrt(2*np.pi*R) * np.exp(-0.5 * (y[n] - g(x_filtre[i]))**2 / R)
    W_filtre /= np.sum(W_filtre)  
    indices = multinomial_resample(W_filtre)
    x_filtre = x_filtre[indices]
    W_filtre = np.ones(N) / N
    x_est = np.sum(W_filtre * x_filtre)
    return x_est, x_filtre, W_filtre

x_part = np.random.normal(0, 1, N)
W_part = np.ones(N) / N
x_estime = np.zeros(T)

for t in range(T):
    x_estime[t], x_part, W_part = filtrage_particulaire_m(x_part, y, W_part, R, N, Q, T, t)

mse = np.mean((x - x_estime)**2)
print(f'Erreur quadratique moyenne: {mse}')

plt.figure(figsize=(10, 6))
plt.plot(x, label='Trajectoire réelle', color='blue')
plt.plot(y, label='Trajectoire observée', color='green', linestyle='dashed')
plt.plot(x_estime, label='Trajectoire estimée', color='red')
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.title('Trajectoires réelle, observée et estimée')
plt.show()

