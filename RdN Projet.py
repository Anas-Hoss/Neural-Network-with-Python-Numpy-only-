# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:07:49 2021

@author: hossi
"""
#Import des packages qu'on va utiliser

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
# Cette fonction sera utile pour la prévision de la fonction qui va partager notre jeu de données
def seuil_de_decision(fonction_decision):
    # Définir les valeurs min max
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Générer une grille de points avec une distance h entre eux
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Prévoir la fonction de décision pour ses points
    Z = fonction_decision(np.c_[xx.ravel(), yy.ravel()])# La fonction ravel() de Numpy rassemble deux arrays en un seul : exemple :x = np.array([[1, 2, 3], [4, 5, 6]]) np.ravel(x) array([1, 2, 3, 4, 5, 6])
    print(Z.shape, xx.shape)
    Z = Z.reshape(xx.shape)# donne une nouvelle taille à Z 
    # Afficher les contours
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Régularisation des paramètres d'affichage
matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)
# Générer un jeu de données et l'afficher (Comme l'exemple de playground TensorFlow)
np.random.seed(0)
X, y = sklearn.datasets.make_moons(n_samples=1000, random_state=42, noise=0.1)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# Voici un autre jeu de données disponible sur la même bibliothèque pour s'amuser 
#X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

taille_entrainement = len(X) # Taille des données d'entrainement
dimension_entree = 2 # la dimension pour l'entrée de la couche
dimension_sortie = 2 # la dimension pour la sortie de la couche

# Paramètres de descente de gradient
alpha = 0.01 # taux d'apprentissage pour la descente de gradient
taux_regularisation = 0.01 # taux de régularisation optionnel

# Pour prévoir la sortie (0 ou 1)
def predict(modele, x):
    W1, b1, W2, b2 = modele['W1'], modele['b1'], modele['W2'], modele['b2']
    # Propagation avant
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1) # Fonction d'activation tanh : (Inspiré de TensorFlow) afin de pouvoir définir la distribution de probabilité
    z2 = a1.dot(W2) + b2
    score_sortie = np.exp(z2)
    prediction_probabiliste = score_sortie / np.sum(score_sortie, axis=1, keepdims=True) # C'est l'implémentation de la fonction softmax qui représente une distribution de probabilité qui nous sera utile lors de la prédiction
    return np.argmax(prediction_probabiliste, axis=1) 


# Permet à notre réseau de neuronne d'apprendre les paramètres et nous renvoie le modèle
# - nombre_noeuds_couches_chachees: Représente le nombre de noeuds dans les couches cachées
# - nombre_iterations: Nombre d'itérations pour la descente de gradient
# - print_loss: Si c'est True, elle renvoie le loss pour chaque 1000 itérations (inspiré de playground TensorFlow)
def construire_modele(nombre_noeuds_couches_chachees, nombre_iterations=10000):
    
    # Initialisation des paramètres par des valeurs aléatoires qu'on va faire apprendre
    np.random.seed(0)
    W1 = np.random.randn(dimension_entree, nombre_noeuds_couches_chachees) / np.sqrt(dimension_entree)
    b1 = np.zeros((1, nombre_noeuds_couches_chachees))
    W2 = np.random.randn(nombre_noeuds_couches_chachees, dimension_sortie) / np.sqrt(nombre_noeuds_couches_chachees)
    b2 = np.zeros((1, dimension_sortie))

    # Notre sortie du modèle sera stocké sur:
    modele = {}
    
    # Descente de gradient:
    for i in range(0, nombre_iterations):

        # Propagation avant
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        score_sortie = np.exp(z2)
        prediction_probabiliste = score_sortie / np.sum(score_sortie, axis=1, keepdims=True)

        # Rétropropagation
        derivee3 = prediction_probabiliste
        derivee3[range(taille_entrainement), y] -= 1
        dW2 = (a1.T).dot(derivee3)
        db2 = np.sum(derivee3, axis=0, keepdims=True)
        derivee2 = derivee3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, derivee2)
        db1 = np.sum(derivee2, axis=0)

        # Ajout des termes de régularisation (Optionnels) (inspiré de TensorFlow)
        dW2 += taux_regularisation * W2
        dW1 += taux_regularisation * W1

        # Mise à jour des vecteurs par descente de gradient
        W1 += -alpha * dW1
        b1 += -alpha * db1
        W2 += -alpha * dW2
        b2 += -alpha * db2
        
        # Redonner les paramètres au modèle
        modele = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return modele

# Construction du modèle pour 3 dimensions de couches cachées
modele = construire_modele(3)

# Affichage du seuil de décision
seuil_de_decision(lambda x: predict(modele, x))
plt.title("Limite de décision pour la couche cachée de taille 3")


plt.figure(figsize=(16, 32))
dimensions_couches_chachees = [1, 2, 3, 4, 5, 20, 50]
for i, nombre_noeuds_couches_chachees in enumerate(dimensions_couches_chachees):
    plt.subplot(5, 2, i+1)
    plt.title('couche cachée %d' % nombre_noeuds_couches_chachees)
    modele = construire_modele(nombre_noeuds_couches_chachees)
    seuil_de_decision(lambda x: predict(modele, x))
plt.show()



























