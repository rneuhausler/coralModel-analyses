#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from coralModel import Reef
from coralModel import Organism
import numpy as np
#import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


#Set Parameters
NumberOfRuns = 100
NumberOfSimulations = 100
coralPercent = .33
algaePercent = .33
turfPercent = 1 - coralPercent - algaePercent
r=1.0
d=.4
a=.2
g=.3
y=.75
dt=.01

length = 10
width = 10
NumberOfNodes = length * width
coralCount = np.zeros((NumberOfRuns, NumberOfSimulations))
turfCount = np.zeros((NumberOfRuns, NumberOfSimulations))
algaeCount = np.zeros((NumberOfRuns, NumberOfSimulations))
types = np.zeros((NumberOfRuns, NumberOfNodes, NumberOfSimulations))


for s in range(0,NumberOfSimulations):
    
#Generate square 8x8 reef with randomly assigned types
#0 = Coral, 1 = Turf, 2 = Algae
    Moorea = Reef()
    count = 0
    for i in range(0,length):
        for j in range(0, width):
            U = np.random.choice([0,1,2],
                                 p=[coralPercent, turfPercent, algaePercent])
            node = Organism(type=U, location=[i,j], ID=count)
            Moorea.append(node)
            count = count + 1
    Moorea.generateGraph()
    #NumberOfNodes = count
    
#Run model 
    for n in range(0,NumberOfRuns):
        for i,val in enumerate(Moorea.nodes):
            types[n,i,s] = Moorea.nodes[i].type
        coralCount[n,s] = np.count_nonzero(types[n,:,s] == 0)
        turfCount[n,s] = np.count_nonzero(types[n,:,s] == 1)
        algaeCount[n,s] = np.count_nonzero(types[n,:,s] == 2)
        Moorea.roll(r=r, d=d, a=a, g=g, y=y, dt=dt)

#save data (probably don't want to save, it's huge)
#np.savetxt("modelOutput.csv", types, delimiter=",")


#Plotting inital and last spatial distribution of model runs
fig, (ax, ax2, cax) = plt.subplots(1,3, 
     gridspec_kw={'width_ratios':[1,1, 0.05]})
fig.subplots_adjust(wspace=0.3)
cmap = plt.get_cmap('Pastel1', 3)
im = ax.imshow(np.reshape(types[0,:,0], (-1, length)), cmap=cmap)
im2 = ax2.imshow(np.reshape(types[NumberOfRuns-1,:,0], (-1, length)), cmap=cmap)

ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
cax.set_axes_locator(ip)
fig.colorbar(im, cax=cax, ax=[ax,ax2], ticks=[0,1,2])
ax.set_title("Spatial dist. of Reef Types, t=0, Sim 1")
ax2.set_title("Spatial dist. of Reef Types, t=100, Sim 1")
plt.show()


#Plotting histograms
fig, (ax, ax2, ax3) = plt.subplots(1,3, facecolor = 'w', sharey='row')
x = np.arange(NumberOfRuns)
im = ax.bar(x, coralCount.mean(axis=1), color='pink')
im2 = ax2.bar(x, turfCount.mean(axis=1), color='navajowhite')
im2 = ax3.bar(x, algaeCount.mean(axis=1), color='gainsboro')
ax.set_ylabel("Average Count over 100 Simulations")
ax.set_xlabel("Time")
ax2.set_xlabel("Time")
ax3.set_xlabel("Time")
ax.set_facecolor('white')
ax2.set_facecolor('white')
ax3.set_facecolor('white')


#Plotting progression over time for 1 simulation
fig1 = plt.figure()
x = np.arange(NumberOfRuns)
im = plt.plot(x, coralCount[:,1], color='pink')
im2 = plt.plot(x, turfCount[:,1], color='navajowhite')
im2 = plt.plot(x, algaeCount[:,1], color='gainsboro')
plt.legend(['Coral', 'Turf', 'Algae'], loc='upper left', fontsize = 'medium')
plt.ylabel("Percent")
plt.xlabel("Time")
plt.show()







