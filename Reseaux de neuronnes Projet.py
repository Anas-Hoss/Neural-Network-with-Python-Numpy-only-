# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:01:26 2021

@author: hossi
"""
import matplotlib.pyplot as plt
import numpy as np

# Définir les fonctions d'activation :
# Dans notre projet on utilise la fonction relu :
def relu(z):
    return np.maximum(0,z)
def derive_relu(z):
    if z>0:
        return 1
    return 0
f=np.vectorize(relu)
df=np.vectorize(derive_relu)
# Facultatif si on veut faire une activation avec la fonction sigmoid :
def sigmoid(z):
    return 1/(1+np.exp(-z))
def derive_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)
g=np.vectorize(sigmoid)
dg=np.vectorize(derive_sigmoid)
# On trace les deux fonctions
X=np.linspace(-2,2,100)
plt.plot(X,f(X))
plt.plot(X,g(X))
# On définit le coût et ça dérivée :
def cout(x,y):
    return (x-y)**2
def derive_cout(x,y):
    return 2*(x-y)
C=np.vectorize(cout)
dC=np.vectorize(derive_cout)

# Définir l'opérateur produit : Qui nous sera utile pour le calcul des gradients :
def produit_termes_termes(A,B):
    return np.array([[A[i][j]*B[i][j]for j in range(len(A[0]))]for i in range(len(A))])
ptt=np.vectorize(produit_termes_termes)
A=np.array([[1],[2]])
B=np.array([[2],[2]])
np.shape(A)
produit_termes_termes(A,B) # petit exemple pour voir si ça marche bien 
np.shape(produit_termes_termes(A,B)) # pour voir la taille de la matrice en sortie

################################################################################################
def propagation_avant(x,W,b):
 ## Calculons tous d'abord les vecteurs de préactivation et activation
 Z=[0 for i in range(len(b))]
 A=[0 for i in range(len(b)+1)]
 A[0]=x # On définit A0 par nos inputs X
 k=0
 while k <len(b):
    Z[k]=np.dot(W[k],A[k])+b[k] # Produit matriciel des poids et A en ajoutant le biais
    A[k+1]=np.array(f(Z[k])) # La matrice A est calculée par l'application de la fonction "ReLu" à la matrice Z
    k+=1
 yhat=A[-1] # la prédiction "yhat" c'est la dernière valeur calculée de la matrice A
 ## Calculons ensuite le gradient du coût par rapport à W, Z et B
 L=[]
 M=[]
 Ga=dC(yhat,y) # On initialise la matrice Gradient de C par rapport à A par la dérivée du coût (yhat-y)²
 l=len(b)-1
 while l>=0:
     Gz=produit_termes_termes(Ga,df(Z[l])) # Gradient de C par rapport à Z : ici on fait le produit terme à terme 
     Gw=np.dot(Gz,np.transpose(A[l])) # Gradient de C par rapport à W : ici on fait le produit de la matrice  Gz (calculé précédemment) avec la transposé de A
     Gb=Gz # Gradient de C par rapport à B c'est Gz
     L.append(Gw)
     M.append(Gz)
     print("A l'etape :",l+1," Le gradient du cout par rapport à W est :",Gw) # Facultatif: Juste pour voir les valeurs pour chaque étape
     print("A l'etape :",l+1," Le gradient du cout par rapport à Z est :",Gz) # Facultatif: Juste pour voir les valeurs pour chaque étape
     print("A l'etape :",l+1," Le gradient du cout par rapport à b est :",Gb) # Facultatif: Juste pour voir les valeurs pour chaque étape
     Ga=np.dot(np.transpose(W[l]),Gz) # Mise à jour de la matrice de gradient de C par rapport à A 
     print("A l'etape :",l," L'initialisation du gradient du cout par rapport à A est :",Ga) # Facultatif: Juste pour voir les valeurs de mise à jour
     l=l-1
 print("On stocke les valeurs du gradient W pour chaque étape :",L) # On garde toutes les valeurs du Gradient de C par rapport à W
 print("On stocke les valeurs du gradient B pour chaque étape :",M) # On garde toutes les valeurs du Gradient de C par rapport à B
 return Z,A,A[-1],L,M,Gz,Gw,Gb

# Définir les inputs dans ce cas : c'est l'exemple traité au cours

y=1
x=np.array([[1],[-2]])
w1=np.array([[0,-1],[2,-3],[1,-1]])
w2=np.array([[0,1,-1],[2,-2,1]])
w3=np.array([[2,-1]])
W=[w1,w2,w3]
b1=np.array([[0],[ 1],[-1]])
b2=np.array([[1],[-2]])
b3=np.array([[0]])
b=[b1,b2,b3]

# print("La taille de la matrice d'entrée est :",np.shape(x)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice W1 est :",np.shape(w1)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice W2 est :",np.shape(w2)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice W3 est :",np.shape(w3)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice b1 est :",np.shape(b1)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice b2 est :",np.shape(b2)) : Facultatif (pour comprendre les tailles en entrée)
# print("La taille de la matrice b3 est :",np.shape(b3)) : Facultatif (pour comprendre les tailles en entrée)
propagation_avant(x,W,b)

#Stocker les valeurs de sortie pour notre exemple:
R=propagation_avant(np.array([[1],[-2]]),[w1,w2,w3],[b1,b2,b3])
Zs=R[0] # On stock la matrice de préactivation Z
As=R[1] # On stock la matrice d'activation A
yhats=R[2] # On stock la valeur de prédiction yhat
gradient_W=R[3] # On stock la matrice du gradient de C par rapport à W
gradient_B=R[4] # On stock la matrice du gradient de C par rapport à B
#gradient_Gz=R[5] : Facultatif (pour avoir les dernières valeurs du gradient Gz)
#gradient_Gw=R[6] : Facultatif (pour avoir les dernières valeurs du gradient de Gw)
#gradient_Gb=R[7] : Facultatif (pour avoir les dernières valeurs du gradient de Gb)
print("Le vecteur Z est :",Zs,"\n","Le vecteur A est :",As,"\n","Le vecteur de sortie est :",yhats,"\n","Le vecteur Gradient de W pour toutes les étapes est :",gradient_W,"\n","Le vecteur Gradient de B pour toutes les étapes est :",gradient_B)
#np.shape(gradient_W[0]): Facultatif (pour comprendre les tailles des matrices)
#np.shape(W[0]): Facultatif (pour comprendre les tailles des matrices)

################################################################################################

def gradient_descente(alpha=0.01, nombre_iterations=2000):
    model = {}
    # Descente de gradient:
    for i in range(0, nombre_iterations):
       for j in range(len(W)):
        W[j]=W[j]-alpha*gradient_W[len(W)-j-1] # Mise à jour de la matrice W avec un pas de gradient alpha
       for k in range(len(b)):
        b[k]=b[k]-alpha*gradient_B[len(b)-k-1] # Mise à jour de la matrice B avec un pas de gradient alpha   
       model = { 'W1': W[0], 'b1': b[0], 'W2': W[1], 'b2': b[1],'W3':W[2],'b3':b[2]} # La matrice W et B après itérations de la descente de gradient 
    return model
gradient_descente(alpha=0.01, nombre_iterations=2000)

################################################################################################
def fit(self, X, y):

 self.X = X
 self.y = y
 self.poids_entree() # initialisation des poids d'entrée W

 for i in range(self.iterations):
    yhat = self.propagation_avant()
    self.retropropagation(yhat)

################################################################################################
def predict(self, X):
    
    Z1 = X.dot(self.params['W1']) + self.params['b1']
    A1 = self.relu(Z1)
    Z2 = A1.dot(self.params['W2']) + self.params['b2']
    A2 = self.relu(Z2)
    Z3 = A2.dot(self.params['W3']) + self.params['b3']
    pred = self.sigmoid(Z3)
    return np.round(pred)

################################################################################################







