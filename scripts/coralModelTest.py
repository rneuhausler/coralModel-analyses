#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Libraries
from coralModelUpdated import Reef, Organism, Ocean
import numpy as np
import sys
import math
import matplotlib.pyplot as plt 
from multiprocessing import Pool
import pickle
import pandas as pd

## Input Parameters

nProcessors = 4
threshold = 1.45                            
coralPercent = .33
algaePercent = .33
r=1.0
d=.4 
a=.2
y=.75
g=.4

#Time and Grid Settings
NumberOfSimulations = 100
tf, dt= 20, .1 
rows, columns = 15, 15

#Also need blobSize, blobValue
gridOption = 0


## Calculated from Input Parameters

turfPercent = 1 - coralPercent - algaePercent
NumberOfTimesteps = int(tf/dt)
NumberOfNodes = rows * columns

## Grid Options Setup
if gridOption == 1: #setup checkered
    m = rows + 2
    n = columns + 2
    checkerBoard = np.tile(np.array([[0,1,2],[1,2,0],[2,0,1]]), ((m+2)//3, (n+2)//3))
    
elif gridOption == 2: #setup blob
    
    blobSize = 5 #input system argument
    blobValue = 0 #input system argument
    notBlob = [a for a in [0,1,2] if a != blobValue]
    
    center = (rows/2, columns/2)
    distanceGrid = np.array([Reef.distance([i+.5,j+.5], center)
                             for i in range(0,rows)
                             for j in range(0,columns)]).reshape(rows,columns)
    blobLocations = (np.where(distanceGrid < blobSize))
    blobLocations = [(blobLocations[0][n],blobLocations[1][n]) 
                      for n in range(0,len(blobLocations[0]))]

data = np.zeros((NumberOfTimesteps, NumberOfNodes))

## Model Run Function

def runModel(simulation):
    print('running simulation' + str(simulation))
    Moorea = Reef()
    count = 0
    
    ## Creating the reef
    for i in range(0,rows):
        for j in range(0,columns):
            
            if gridOption == 0:
                U = np.random.choice([0,1,2],
                                     p=[coralPercent, turfPercent, algaePercent])
            elif gridOption == 1:
                U = checkerBoard[i,j]
            
            elif gridOption == 2:
                if (i,j) in blobLocations: 
                    U = blobValue
                else:
                    U = np.random.choice(notBlob,
                                         p=[.5, .5])
            node = Organism(type=U, location=[i,j], ID=count)
            Moorea.append(node)
            count = count + 1
    Moorea.generateGraph(threshold)
    
    ## Running new reef through time
    for n in range(0,NumberOfTimesteps):
        for i,val in enumerate(Moorea.nodes):
            data[n,i] = Moorea.nodes[i].type
        Moorea.roll(r=r, d=d, a=a, g=g, y=y, dt=dt)
    
    if simulation = 0: ## check on progress
        plt.figure()
        plt.imshow(np.reshape(np.array([Moorea.nodes[i].type for
                                        i,val in enumerate(Moorea.nodes)]),(rows,columns)))
    return(data)
 
## Run Model
    
if __name__ == '__main__':
        
    with Pool(nProcessors) as p:
        results = p.map(runModel, np.arange(NumberOfSimulations)) ## function, argument

## Save Output File

path = 'outputFolder/'
output_name = 'parameterValues'
outfile = open(path+output_name, "wb") #open pickle jar
pickle.dump(results, outfile)           #put contents into jar
outfile.close()                        #close the jar

## Pull some information (using parallelization) and save into pandas table

## save csv files
path = 'outputFolder/'
output_name = 'summary'
np.savetxt(path+ output_name + '.csv', types, delimiter=",")





