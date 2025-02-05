import numpy as np
import matplotlib.pyplot as plt
import time  # Pour mesurer le temps d'exécution
from numpy.random import random

def f(x_prec, n):
    return 0.5*x_prec + 25*x_prec/(1 + x_prec**2) + 8*np.cos(1.2*n)

def creation_trajectoire(x0, Q, T):
    x = np.zeros(T)
    x[0] = x0
    for i in range(1, T):
        x[i] = f(x[i-1], i) + np.random.normal(0, Q)
    return x

def g(x):
    return x**2 / 20

def creation_observation(x, R):
    y = np.zeros(T)
    for i in range(T):
        y[i] = g(x[i]) + np.random.normal(0, R)
    return y

def multinomial_resample(weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.
    return np.searchsorted(cumulative_sum, random(len(weights)))

def filtrage_particulaire_m(x_part, y, W_part, R, N, Q, n):
    x_filtre = np.zeros(N)
    W_filtre = np.zeros(N)
    for i in range(N):
        x_filtre[i] = f(x_part[i], n) + np.random.normal(0, Q)
        W_filtre[i] = W_part[i] * 1/np.sqrt(2*np.pi*R) * np.exp(-0.5 * (y[n] - g(x_filtre[i]))**2 / R)
    W_filtre /= np.sum(W_filtre)
    indices = multinomial_resample(W_filtre)
    x_filtre = x_filtre[indices]
    W_filtre = np.ones(N) / N
    x_est = np.sum(W_filtre * x_filtre)
    return x_est, x_filtre, W_filtre

# Paramètres du problème
x0 = 0
Q = 10
T = 50  # Nombre de points dans la trajectoire
R = 1

# Création de la trajectoire réelle et des observations
x = creation_trajectoire(x0, Q, T)
y = creation_observation(x, R)

# Valeurs de N à tester
N_values = [10, 50, 100, 500, 1000]  # Différents nombres de particules
mse_values = []  # Stockage des erreurs quadratiques moyennes
execution_times = []  # Stockage des temps d'exécution

# Boucle pour tester différents N
for N in N_values:
    # Initialisation des particules et des poids
    x_part = np.random.normal(0, 1, N)
    W_part = np.ones(N) / N
    x_estime = np.zeros(T)

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Filtrage particulaire pour ce N
    for t in range(T):
        x_estime[t], x_part, W_part = filtrage_particulaire_m(x_part, y, W_part, R, N, Q, t)

    # Calcul de l'erreur quadratique moyenne
    mse = np.mean((x - x_estime)**2)
    mse_values.append(mse)

    # Enregistrer le temps d'exécution
    end_time = time.time()
    execution_times.append(end_time - start_time)

    print(f'N = {N}, Erreur quadratique moyenne: {mse}, Temps d\'exécution: {end_time - start_time:.4f} secondes')

# Tracer les résultats
plt.figure(figsize=(12, 5))

# Courbe de l'EQM
plt.subplot(1, 2, 1)
plt.plot(N_values, mse_values, marker='o', color='blue', linestyle='--')
plt.xlabel('Nombre de particules N')
plt.ylabel('Erreur quadratique moyenne (EQM)')
plt.title('EQM en fonction de N')
plt.grid(True)

# Courbe du temps d'exécution
plt.subplot(1, 2, 2)
plt.plot(N_values, execution_times, marker='o', color='red', linestyle='--')
plt.xlabel('Nombre de particules N')
plt.ylabel('Temps d\'exécution (secondes)')
plt.title('Temps d\'exécution en fonction de N')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, label='Trajectoire réelle', color='blue')
plt.plot(y, label='Trajectoire observée', color='green', linestyle='dashed')
plt.plot(x_estime, label='Trajectoire estimée', color='red')
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.title('Trajectoires réelle, observée et estimée')
plt.show()

