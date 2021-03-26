#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from coralModel import Reef, Organism, Ocean
import tools as tl
import numpy as np
import sys
import pandas as pd
from multiprocessing import Pool


##  Parameters

number_of_processors = int(sys.argv[1])
number_of_simulations = int(sys.argv[2])

coral_percent = float(sys.argv[3])/100
macroalgae_percent = float(sys.argv[4])/100
turf_percent = round(1 - coral_percent - macroalgae_percent, 2)

grid_option = int(sys.argv[5])
number_of_rows, number_of_columns = int(sys.argv[6]), int(sys.argv[7])
neighborhood_threshold = float(sys.argv[8])
record_rate = int(sys.argv[9])
image_return = str(sys.argv[10]) ## works, but may want to figure out how to make into boolean
image_record_rate = int(sys.argv[11])
image_counter = int(sys.argv[11])
r, d, a, g, y, dt, tf = [float(sys.argv[n]) for n in range(12,19)]

number_of_timesteps = int(tf/dt)
number_of_recorded_timesteps = round(number_of_timesteps / record_rate)
number_of_nodes = number_of_rows * number_of_columns

if grid_option == 1:
    checker_board = tl.generate_checker_board(number_of_rows, number_of_columns)

elif grid_option == 2:
    type_percent_dictionary = {0:coral_percent, 1:turf_percent, 2:macroalgae_percent}
    blob_value = int(sys.argv[19])
    blob_percent = type_percent_dictionary[blob_value]
    not_blob = [a for a in [0,1,2] if a != blob_value]
    not_blob_percentages_adjusted = [round(type_percent_dictionary[n]/
                                           (1 - blob_percent), 2) for n in not_blob]

    blob_locations = tl.generate_blob(blob_percent, blob_value,
                                      number_of_rows, number_of_columns,
                                      number_of_nodes)
##  Functions

def create_reef(simulation):

    Moorea = Reef()
    count = 0

    if grid_option == 0:

        grid = [(i,j)
                for i in range(0, number_of_columns)
                for j in range(0, number_of_rows)]
        np.random.shuffle(grid)
        coral_count = round(number_of_nodes*coral_percent)
        macro_count = round(number_of_nodes*macroalgae_percent)
        locations = {'coral': grid[0:coral_count],
                     'macro': grid[coral_count: coral_count + macro_count]}

    for i in range(0, number_of_rows):

        for j in range(0, number_of_columns):

            if grid_option == 0:
                if (i,j) in locations['coral']:
                    U = 0
                elif (i,j) in locations['macro']:
                    U = 2
                else:
                    U = 1
            ## old random grid (issue: inconsistent initial percentages)
            #    U = np.random.choice([0,1,2],  p=[coral_percent, turf_percent, macroalgae_percent])

            elif grid_option == 1:
                U = checker_board[i,j]

            elif grid_option == 2:
                if (i,j) in blob_locations:
                    U = blob_value
                else:
                    U = np.random.choice(not_blob,
                                         p=not_blob_percentages_adjusted)
            node = Organism(type=U, location=[i,j], ID=count)
            Moorea.append(node)
            count = count + 1
    return(Moorea)


def pull_info(Moorea, simulation, timestep, image_counter=image_counter):
    image = np.array([Moorea.nodes[n].type
                      for n,val in enumerate(Moorea.nodes)])
    coral_count = np.count_nonzero(image==0)
    coral_neighbors = tl.extract_neighbors(Moorea, int(0), coral_count)
    turf_count = np.count_nonzero(image==1)
    turf_neighbors = tl.extract_neighbors(Moorea, int(1), turf_count)
    macroalgae_count = np.count_nonzero(image==2)
    macroalgae_neighbors = tl.extract_neighbors(Moorea, int(2), macroalgae_count)
    coral_patch_count, algae_patch_count, turf_patch_count, macroalgae_patch_count = tl.patch_counts(image, number_of_rows)
    data = [simulation, timestep, coral_count, turf_count, macroalgae_count,
            coral_neighbors, turf_neighbors, macroalgae_neighbors,
            coral_patch_count, algae_patch_count, turf_patch_count, macroalgae_patch_count, image]
    dataframe = pd.DataFrame([data])

    if image_return == 'true':
        if image_counter == image_record_rate:
            np.savetxt(path+'images/'+str(simulation)+'_'+str(timestep)+'_'+
                       name+'.csv', image.astype(int), delimiter=',')

            image_counter = 0
        image_count = image_counter + 1
    return(dataframe)


def run_model(simulation):

    np.random.seed(simulation)

    print('running simulation' + str(simulation))

    Moorea = create_reef(simulation)
    Moorea.generate_graph(neighborhood_threshold)

    for timestep in range(0, number_of_timesteps):
        if timestep == 0:
            table = pd.DataFrame([])
        if timestep % record_rate == 0:
            table = pd.concat([table, pull_info(Moorea, simulation, timestep)])

        Moorea.roll(r=r, d=d, a=a, g=g, y=y, dt=dt)

    return(table)

## Run

if __name__ == '__main__':

    path='./output/'+str(number_of_rows)+'x'+str(number_of_columns)+'/grid'+str(grid_option)+'/grazing'+str(round(g*100))+'/threshold'+str(int(neighborhood_threshold*100))+'/'

    name='coral'+str(int(coral_percent*100))+'-macro'+str(int(macroalgae_percent*100))+'-r'+str(int(r*10))+'-d'+str(int(d*100))+'-a'+str(int(a*100))+'-y'+str(int(y*100))+'-time'+str(number_of_timesteps)+'-rec'+str(record_rate)+'-nsim'+str(number_of_simulations)

    with Pool(number_of_processors) as p:
        output = pd.concat(p.map(run_model, np.arange(number_of_simulations)))

    output.columns = ['simulation', 'timestep', 'coral_count', 'turf_count',
                      'macroalgae_count', 'coral_neighbors', 'turf_neighbors', 'macroalgae_neighbors',
                      'coral_patch_count', 'algae_patch_count', 'turf_patch_count', 'macroalgae_patch_count', 'image']

    output.to_csv(path+name+'.csv', header=True, index=False)
