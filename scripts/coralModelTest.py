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
if gridOption == 1:
    checkerBoard = tl.generateCheckerBoard(rows, columns)
elif gridOption == 2:
    blobLocations, notBlob = tl.generateBlob(coralPercent,
                                              blobValue =
                                              int(sys.argv[18])) 

##  Functions


def createReef():
    Moorea = Reef()
    count = 0   
    for i in range(0,rows):
        for j in range(0,columns):
            
            if gridOption == 0:
                U = np.random.choice([0,1,2], 
                                     p=[coralPercent,
                                        turfPercent, 
                                        algaePercent])
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


def pullInfo(Moorea, simulation, timestep):
    image = np.array([Moorea.nodes[n].type
                      for n,val in enumerate(Moorea.nodes)])
    C = np.count_nonzero(image==0)
    CN = round(tl.densityExtract(Moorea, int(0), C), 2)
    T = np.count_nonzero(image==1)
    TN = round(tl.densityExtract(Moorea, int(1), T), 2)
    M = np.count_nonzero(image==2)
    MN = round(tl.densityExtract(Moorea, int(2), M), 2)
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


## Run


if __name__ == '__main__':
       
    with Pool(nProcessors) as p:
        output = pd.concat(p.map(runModel, np.arange(NumberOfSimulations)))     ##
        
    output.columns = ['Simulation', 'Timestep', 'CoralCount', 'TurfCount',
                      'MacroalgaeCount', 'Coral-CoralNeighbors', 
                      'Turf-TurfNeighbors', 'Macro-MacroNeighbors',
                      'CoralPatchCount', 'AlgaePatchCount', 'TurfPatchCount',
                      'MacroPatchCount']
    
    output.to_csv('table.csv', header=True, index=False)
