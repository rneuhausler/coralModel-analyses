#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from coralModel import Reef, Organism, Ocean
import tools as tl
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool

##  Parameters

nProcessors = int(sys.argv[1])
NumberOfSimulations = int(sys.argv[2])
coralPercent = float(sys.argv[3])
algaePercent = float(sys.argv[4])
gridOption = int(sys.argv[5])
rows, columns = int(sys.argv[6]), int(sys.argv[7])
threshold = float(sys.argv[8])
recordRate = int(sys.argv[9])
r, d, a, y, g, dt, tf = [float(sys.argv[n]) for n in range(10,17)]
NumberOfTimesteps = int(tf/dt)
NumberOfRecordedTimesteps = round(NumberOfTimesteps / recordRate)
NumberOfNodes = rows * columns
turfPercent = 1 - coralPercent - algaePercent


##  Functions

def generateCheckerBoard():
    m = rows + 2
    n = columns + 2
    checkerBoard = np.tile(np.array([[0,1,2],[1,2,0],[2,0,1]]), 
                           ((m+2)//3, (n+2)//3))
    return(checkerBoard)

def generateBlob(blobPercent):
    blobValue = int(sys.argv[18])
    notBlob = [a for a in [0,1,2] if a != blobValue]   
    center = (rows/2, columns/2)
    distanceGrid = np.array([Reef.distance([i+.5,j+.5], center)
                             for i in range(0,rows)
                             for j in range(0,columns)])
    maxDistance = np.sort(distanceGrid)[round(blobPercent*NumberOfNodes)]
    blobLocations = (np.where(distanceGrid.reshape(rows,columns) < maxDistance))
    blobLocations = [(blobLocations[0][n],blobLocations[1][n]) 
                     for n in range(0,len(blobLocations[0]))]
    return(blobLocations, notBlob)
    
def createReef():
    Moorea = Reef()
    count = 0   
    for i in range(0,rows):
        for j in range(0,columns):
            
            if gridOption == 0:
                U = np.random.choice([0,1,2], p=[coralPercent, 
                                                 turfPercent, algaePercent])
            elif gridOption == 1:
                U = checkerBoard[i,j]
                
            elif gridOption == 2:
                if (i,j) in blobLocations: 
                    U = blobValue
                else:
                    U = np.random.choice(notBlob, p=[.5, .5])
            node = Organism(type=U, location=[i,j], ID=count)
            Moorea.append(node)
            count = count + 1
    return(Moorea)

def densityExtract(Moorea, Type, Count):
    if Count == 0:
        neighbors = 0
    else:
        neighbors = np.array([Moorea.nodes[n].density/
                              Moorea.nodes[n].density.sum()
                              for n,val in enumerate(Moorea.nodes)
                              if Moorea.nodes[n].type == Type]).mean(axis=0)[Type]
    return(neighbors)

def pullInfo(Moorea, simulation, timestep):
    image = np.array([Moorea.nodes[n].type
                      for n,val in enumerate(Moorea.nodes)])
    C = np.count_nonzero(image==0)
    CN = densityExtract(Moorea, int(0), C)
    T = np.count_nonzero(image==1)
    TN = densityExtract(Moorea, int(1), T)
    M = np.count_nonzero(image==2)
    MN = densityExtract(Moorea, int(2), M)
    CP, AP, TP, MP = tl.patchCounts(image, rows)
    data = [simulation, timestep, C, T, M, CN, TN, MN, CP, AP, TP, MP]  
    dataframe = pd.DataFrame([data])
    return(dataframe)

def runModel(simulation):
    
    print('running simulation' + str(simulation))
    
    Moorea = createReef()
    Moorea.generateGraph(threshold) 
    
    for timestep in range(0,NumberOfTimesteps):
        if timestep == 0:
            table = pd.DataFrame([])
        if timestep % recordRate == 0:
            table = pd.concat([table, pullInfo(Moorea, simulation, timestep)])      ## 
            
        Moorea.roll(r=r, d=d, a=a, g=g, y=y, dt=dt)
        
    return(table)

## Grid Options Setup

if __name__ == '__main__':
    
    if gridOption == 1:
        checkerBoard = generateCheckerBoard(rows, columns)
    elif gridOption == 2:
        blobLocations, notBlob = generateBlob(coralPercent)    
    
    with Pool(nProcessors) as p:
        output = pd.concat(p.map(runModel, np.arange(NumberOfSimulations)))     ##
        
    output.columns = ['Simulation', 'Timestep', 'CoralCount', 'TurfCount',
                      'MacroalgaeCount', 'Coral-CoralNeighbors', 
                      'Turf-TurfNeighbors', 'Macro-MacroNeighbors',
                      'CoralPatchCount', 'AlgaePatchCount', 'TurfPatchCount',
                      'MacroPatchCount']
    
    output.to_csv('table.csv', header=True, index=False)