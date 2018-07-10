# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:31:36 2017

@author: wilms
"""

import numpy as np

def vergleicheHist(mode, h1, h2): #ueber den Parameter mode laesst sich zwischen euklidischer Distanz und Intersection auswaehlen
    if mode == 'i':
        return 1-np.sum(np.minimum(h1,h2))/float(np.sum(h1))#Formel für die Intersection siehe Folie 22, np.minimum ermittelt die paarweisen Minima zweier gleich geformter Arrays
    if mode == 'e':
        return np.linalg.norm(h1-h2)#numpy-Funktion zur Berechnung der euklidischen Distanz, sofern die Eingabe ein Array mit nur einer Dimension ist
    
    
def getMerkmal(img, feature,n):
    if feature == 'mean':
        return np.mean(img)
    if feature == 'std':
        return np.std(img)
    if feature == 'mean+std':
        return np.array([np.mean(img),np.std(img)]) 
    if feature == 'histogram1D':
        return np.histogram(img, bins = n, range=(0,256))[0]

d = np.load('./trainingsDaten2.npz')
trImgs = d['data']
trLabels = d['labels']

d = np.load('./validierungsDaten2.npz')
vaImgs = d['data']
vaLabels = d['labels']

n = 15

trMerkmale = [] #leere Listen, in die die Merkmale abgelegt werden
vaMerkmale = [] #zwei Listen pro Bild, um mehrere Merkmale miteinander gewichten zu können
trMerkmale2 = [] 
vaMerkmale2 = []
for img in trImgs: #fuer jedes Trainingsbild einen Deskriptor berechnen und diesen an die Liste anhaengen
    trMerkmale.append(getMerkmal(img, 'mean+std',n))
    trMerkmale2.append(getMerkmal(img, 'std',0))
    
for img in vaImgs:
    vaMerkmale.append(getMerkmal(img, 'mean+std',n))
    vaMerkmale2.append(getMerkmal(img, 'std',0))
    

W = 0
result = []

for vaM,vaM2 in zip(vaMerkmale,vaMerkmale2): #fuer jedes Validierungsbild bzw. dessen Merkmale
    distances = [] #initialisiere eine Liste, die die Distanzen zwischen einem Validierungsbild und den verschiedenen Trainingsbildern aufnimmt
    for trM,trM2 in zip(trMerkmale,trMerkmale2): #fuer jedes Trainingsbild bzw. dessen Merkmale
        distances.append(np.linalg.norm(vaM-trM)+W*np.linalg.norm(vaM2-trM2))#berechne euklidische Distanz zwischen den Merkmalem (ggf. gewichtet) und haenge sie an die Liste an
#        distances.append(vergleicheHist('e',vaM,trM)) #Vergleich fuer Histogramme
    result.append(trLabels[np.argmin(distances)]) #argmin gibt den Index der kleinsten Distanz zurueck, der dem Index des am besten passenden Trainingsbildes entspricht
    #mit dem Index kann das Labels dieses Trainingsbildes geholt werden, was der Vorhersage entspricht

correct = 0.0
for vorhersage, groundTruth in zip(result,vaLabels): #fuer jede vorhersage und jedes wirkliche Label
    if vorhersage == groundTruth: 
        correct+=1 #zahle die korrekten Zuordnungen
        
acc = correct/len(vaLabels) #hier muss einer der beiden Teile ein float sein, sonst bekommt man nur 0 oder 1

print n,W, acc

confusionMatrix = np.zeros((3,3))  #erstellt ein Array der Groesse 3x3 mit Nullen gefuellt
for vorhersage, groundTruth in zip(result,vaLabels): 
    indexV = int(vorhersage/4) #Ganzzahldivision durch 4, damit Label 1 auf Index 0, Label 4 auf Index 4 und Label 8 auf Index 2 abgebildet wird
    indexGT = int(groundTruth/4)
    confusionMatrix[indexGT,indexV]+=1
print confusionMatrix