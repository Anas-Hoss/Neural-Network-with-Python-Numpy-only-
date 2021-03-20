# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:55:46 2021

@author: hossi
"""
import numpy as np 
import matplotlib.pyplot as plt

######Définir les fonctions d'activation##########
def relu(z):
    return np.maximum(0,z)
def derive_relu(z):
    if z>0:
        return 1
    return 0
f=np.vectorize(relu)
df=np.vectorize(derive_relu)

def sigmoid(z):
    return 1/(1+np.exp(-z))
def derive_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)
g=np.vectorize(sigmoid)
dg=np.vectorize(derive_sigmoid)

X=np.linspace(-2,2,100)
plt.plot(X,f(X))
plt.plot(X,g(X))

######### Définir l'opérateur produit##########
def produit_termes_termes(A,B):
    return np.array([[A[i][j]*B[i][j]for j in range(len(A[0]))]for i in range(len(A))])
ptt=np.vectorize(produit_termes_termes)
A=np.array([[1],[2]])
B=np.array([[2],[2]])
np.shape(A)
produit_termes_termes(A,B)
np.shape(produit_termes_termes(A,B))
######## Fonction Coût###############
def cout(x,y):
    return (x-y)**2
def derive_cout(x,y):
    return 2*(x-y)
C=np.vectorize(cout)
dC=np.vectorize(derive_cout)

########Définition des données d'entrée###########
y=1
inputs=np.array([[1],[-2]])
print("La taille de la matrice d'entrée est :",np.shape(inputs))
W1=np.array([[0,-1],[2,-3],[1,-1]])
print("La taille de la matrice W1 est :",np.shape(W1))
W2=np.array([[0,1,-1],[2,-2,1]])
print("La taille de la matrice W2 est :",np.shape(W2))
W3=np.array([[2,-1]])
print("La taille de la matrice W3 est :",np.shape(W3))
W=[W1,W2,W3]
b1=np.array([[0],[ 1],[-1]])
print("La taille de la matrice b1 est :",np.shape(b1))
b2=np.array([[1],[-2]])
print("La taille de la matrice b2 est :",np.shape(b2))
b3=np.array([[0]])
print("La taille de la matrice b3 est :",np.shape(b3))
b=[b1,b2,b3]
Z=[0 for i in range(len(b))]
A=[inputs,0,0,0]

########## Matrice d'activation et préactivation############
k=0
while k <len(b):
    Z[k]=np.dot(W[k],A[k])+b[k]
    A[k+1]=np.array(f(Z[k]))
    k+=1
yhat=A[-1]
print("La matrice de préactivation est : ",Z)
print("La matrice d'activation est : ",A)
print("La prédiction est : ",yhat)
Zs=Z
As=A
print(Zs)
print(As)

##### Calcul de la matrice dérivée de Relu #########
for t in range (len(Zs)):
    Zs[t]=df(Zs[t])
Zr=Zs
print(Zr)
print(As)
##### Algo de rétropropagation#########
L=[]
M=[]
Ga=dC(yhat,y)
l=len(b)-1
while l>=0:
    Gz=produit_termes_termes(Ga,df(Zr[l]))
    Gw=np.dot(Gz,np.transpose(As[l]))
    Gb=Gz
    L.append(Gw)
    M.append(Gz)
    print("A l'etape :",l+1," Le gradient du cout par rapport à Z est :",Gz)
    print("A l'etape :",l+1," Le gradient du cout par rapport à W est :",Gw)
    print("A l'etape :",l+1," Le gradient du cout par rapport à b est :",Gb)
    Ga=np.dot(np.transpose(W[l]),Gz)
    print("A l'etape :",l," L'initialisation du gradient du cout par rapport à A est :",Ga)
    l=l-1
print("On stocke les valeurs du gradient W pour chaque étape :",L)
print("On stocke les valeurs du gradient B pour chaque étape :",M)
########################################
a=0.02
W1=W1-a*L[0]
W2=W2-a*L[1]
W3=W3-a*L[2]
b1=b1-a*M[0]
b2=b2-a*M[1]
b3=b3-a*M[2]
W=[W1,W2,W3]
b=[b1,b2,b3]
print(W,"fzefergfer",b)
k=0
while k < len(Z):
 Z[k]=np.dot(W[k],A[k])+b[k]
 print(k,Z[k])
 A[k+1]=np.array(f(Z[k]))
 print(k+1,A[k])
 k+=1
print("fdgrere",A[-1])
#######################################













