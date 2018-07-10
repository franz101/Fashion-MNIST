# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:31:36 2017

@author: wilms
"""

import numpy as np

def vergleicheHist(mode, h1, h2): 
    if mode == 'i':
        return 1-np.sum(np.minimum(h1,h2))/float(np.sum(h1))
    if mode == 'e':
        return np.linalg.norm(h1-h2) 
        
def getMerkmal(img, feature,n):
    if feature == 'mean':
        return np.mean(img, axis=(0,1))
    if feature == 'std':
        return np.std(img, axis=(0,1))
    if feature == 'histogram1D':
        rHist = np.histogram(img[:,:,0], bins = n, range=(0,256))[0] 
        gHist = np.histogram(img[:,:,1], bins = n, range=(0,256))[0]
        bHist = np.histogram(img[:,:,2], bins = n, range=(0,256))[0]
        return np.hstack((rHist, gHist, bHist)) #hstack verbindet die Eingabe (hier die einzelnen Historgamme als Array) zu einem Array, indem es die Elemente horizontal stapelt
    if feature == 'histogram3D':
        imgReshaped = img.reshape((img.shape[0]*img.shape[1],3)) #Reshapen, damit jedes Pixek in einer Zeile liegt
        return np.histogramdd(imgReshaped, bins = [n,n,n], range=((0,256),(0,256),(0,256)))[0].flatten()

d = np.load('./trainingsDatenFarbe2.npz')
trImgs = d['data']
trLabels = d['labels']

d = np.load('./validierungsDatenFarbe2.npz')
vaImgs = d['data']
vaLabels = d['labels']

trMerkmale = [] 
vaMerkmale = []
trMerkmale2 = [] 
vaMerkmale2 = []
n=7
for img in trImgs: 
    trMerkmale.append(getMerkmal(img, 'histogram3D', n))
    trMerkmale2.append(getMerkmal(img, 'std', 0))
    
for img in vaImgs:
    vaMerkmale.append(getMerkmal(img, 'histogram3D', n))
    vaMerkmale2.append(getMerkmal(img, 'std', 0))

W=0
result = []
for vaM,vaM2 in zip(vaMerkmale,vaMerkmale2):
    distances = [] 
    for trM,trM2 in zip(trMerkmale,trMerkmale2): 
        distances.append(np.linalg.norm(vaM-trM)+W*np.linalg.norm(vaM2-trM2)) 
#            distances.append(vergleicheHist('e',vaM,trM))
    result.append(trLabels[np.argmin(distances)]) 

correct = 0.0
for i,r in enumerate(result): 
    if r == vaLabels[i]: 
        correct+=1
        
acc = correct/len(vaLabels) 

print n,W, acc
