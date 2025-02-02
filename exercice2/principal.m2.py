import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import os

def lecture_image():
    SEQUENCE = "./exercice2/videos_sequences/sequence1/sequence1/"
    filenames = sorted(os.listdir(SEQUENCE))
    T = len(filenames)
    im = Image.open(os.path.join(SEQUENCE, filenames[0]))
    return im, filenames, T, SEQUENCE

def selectionner_zone(im):
    plt.figure("Sélectionnez la zone à suivre")
    plt.imshow(im)
    print('Cliquez 4 points pour définir la zone.')
    
    points = []
    while len(points) < 4:
        pt = plt.ginput(1, timeout=-1)[0]
        plt.plot(pt[0], pt[1], 'rx')
        points.append(pt)
        plt.draw()
    
    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    zoneAT = [x_min, y_min, x_max-x_min, y_max-y_min]
    
    rect = ptch.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                         linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    return np.array(zoneAT)

def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    print(im)
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb
            
    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

  #  print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
   # print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    return (new_im,kmeans,histogramme)

N=100
N_b=100
Lambda=50
C1=3000
C2=3000
Q=np.array([[C1,0],[0,C2]])

def f(x_prec):
    return np.random.multivariate_normal(x_prec,Q,N)

def D(q,q_prime):
    somme=0
    for i in range(N_b):
        somme+=np.sqrt(q[i]*q_prime[i])
    return np.sqrt(1-somme)
    

def multinomial_resample(weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # Force la somme cumulée à terminer à 1.0
    return np.searchsorted(cumulative_sum, np.random.random(len(weights)))

def filtrage_particulaire_m(im, particles_prev, weights_prev, Q, q_ref, zone_ref):
    N = len(particles_prev)
    
    # 1. Propagation des particules
    particles = np.zeros_like(particles_prev)
    for i in range(N):
        particles[i] = np.random.multivariate_normal(particles_prev[i], Q)
    
    # 2. Calcul des poids
    weights = np.zeros(N)
    for i in range(N):
        # Calcul de l'histogramme pour la particule
        x, y = particles[i]
        w, h = zone_ref[2], zone_ref[3]

        _, _, hist = calcul_histogramme(im, [x, y, w, h], q_ref[1])
        d = D(q_ref[2], hist)
        weights[i] = np.exp(-Lambda * d**2)
    
    weights /= weights.sum()
    
    indices = multinomial_resample(weights)
    particles = particles[indices]

    x_est = np.mean(particles, axis=0)
    
    return x_est, particles, weights

# Code principal corrigé
im, filenames, T, SEQUENCE = lecture_image()

# Afficher l'image correctement
plt.imshow(im)
plt.axis('image')
plt.show(block=False)
plt.pause(0.1)

# Sélection de zone
zoneAT = selectionner_zone(im)
plt.close('all')

# Calcul de l'histogramme de référence
_, kmeans_ref, hist_ref = calcul_histogramme(im, zoneAT, N_b)
q_ref = (zoneAT, kmeans_ref, hist_ref)

# Initialisation des particules (position initiale + bruit)
x_initial = np.array([zoneAT[0], zoneAT[1]])
x_part = np.random.multivariate_normal(
    mean=x_initial, 
    cov=[[C1, 0], [0, C2]], 
    size=N
)

# Boucle principale sur les images ---------------------------------
plt.figure(figsize=(10, 6))
for t in range(T):
    # Chargement de l'image
    im = Image.open(os.path.join(SEQUENCE, filenames[t]))
    
    # Filtrage particulaire
    x_est, x_part, W_part = filtrage_particulaire_m(im, x_part, None, Q, q_ref, zoneAT)
    
    # Visualisation
    plt.clf()
    plt.imshow(im)
    
    # Affichage des particules
    plt.scatter(x_part[:,0], x_part[:,1], c='r', s=10, alpha=0.4)
    
    # Affichage du rectangle estimé
    rect = ptch.Rectangle(
        (x_est[0], x_est[1]), 
        zoneAT[2], 
        zoneAT[3],
        linewidth=2, 
        edgecolor='g', 
        facecolor='none'
    )
    plt.gca().add_patch(rect)
    
    plt.title(f"Frame {t+1}/{T}")
    plt.pause(0.01)
    plt.draw()

plt.show()
    

    



