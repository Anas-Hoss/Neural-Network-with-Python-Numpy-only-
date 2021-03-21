# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:03:27 2021

@author: hossi
"""
import numpy as np
class Reseau_Neuronne():
    def propagation_avant(self,x,W,b):      
        ## Calculons tous d'abord les vecteurs de préactivation et activation
        def relu(z):
          return np.maximum(0,z)
        f=np.vectorize(relu)
        Z=[0 for i in range(len(b))]
        A=[0 for i in range(len(b)+1)]
        A[0]=x # On définit A0 par nos inputs X
        k=0
        while k <len(b):
          Z[k]=np.dot(W[k],A[k])+b[k] # Produit matriciel des poids et A en ajoutant le biais
          A[k+1]=np.array(f(Z[k])) # La matrice A est calculée par l'application de la fonction "ReLu" à la matrice Z
          k+=1
        yhat=A[-1]   
        return yhat # la prédiction "yhat" c'est la dernière valeur calculée de la matrice A

    def retro_propagation(self,x,y,W,b,alpha):
        # Définir la fonction d'activation : Relu et sa dérivée
        def relu(z):
          return np.maximum(0,z)
        def derive_relu(z):
            if z>0:
             return 1
            return 0
        f=np.vectorize(relu)
        df=np.vectorize(derive_relu)
        
        # Définir la fonction coût et sa dérivée
        def cout(x,y):
          return (x-y)**2
        def derive_cout(x,y):
          return 2*(x-y)
        dC=np.vectorize(derive_cout)

        # Définir l'opérateur produit : Qui nous sera utile pour le calcul des gradients :
        def produit_termes_termes(A,B):
          return np.array([[A[i][j]*B[i][j]for j in range(len(A[0]))]for i in range(len(A))])

        Z=[0 for i in range(len(b))]
        A=[0 for i in range(len(b)+1)]
        A[0]=x # On définit A0 par nos inputs X
        k=0
        while k <len(b):
          Z[k]=np.dot(W[k],A[k])+b[k] # Produit matriciel des poids et A en ajoutant le biais
          A[k+1]=np.array(f(Z[k])) # La matrice A est calculée par l'application de la fonction "ReLu" à la matrice Z
          k+=1
        yhat=A[-1]
        
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
        
        #Mise à jour des poids et des biais
        for j in range(len(W)):
          W[j]=W[j]-alpha*L[len(W)-j-1] # Mise à jour de la matrice W avec un pas de gradient alpha
        for k in range(len(b)):
          b[k]=b[k]-alpha*M[len(b)-k-1] # Mise à jour de la matrice B avec un pas de gradient alpha   
        return " le vecteur W après mise à jour est :" ,W," le vecteur b après mise à jour est :" ,b

rn=Reseau_Neuronne()
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
rn.propagation_avant(x, W, b)
rn.retro_propagation(x, y, W, b, alpha=0.01)

