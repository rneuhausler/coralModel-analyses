import numpy as np
import math


class Organism():

    def __init__(self, ID, type, location):
        self.ID = ID
        self.type = type
        self.neighbors = np.zeros(3)
        self.location = location

class Reef():
        # append(): Add organisms to create a Reef

        # generate_graph(): Creates a graph of appended nodes of class Organism,
        # based on their defined 2-d coordinate location

        # roll(): one stochastic update of each node, weighted odds based on
        # differential equation in documentation

    def __init__(self):
        self.nodes = []
        self.graph = {}
        self.updates = {}
        #self.configs = {}

    def append(self, node):
        self.nodes.append(node)

    def distance(a,b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx*dx + dy*dy)

    def count_neighbor_types(self):
        for i,val in enumerate(self.nodes):
            for n in self.graph[i]:
                if self.nodes[n].type == 0:
                    self.nodes[i].neighbors[0] += 1
                elif self.nodes[n].type == 1:
                    self.nodes[i].neighbors[1] += 1
                elif self.nodes[n].type == 2:
                    self.nodes[i].neighbors[2] += 1

    def generate_graph(self, threshold=1.5):
        self.graph = {self.nodes[i].ID:list(self.nodes[j].ID
                      #can add things, e.g. self.nodes[j].type
                                            for j,val in enumerate(self.nodes)
                                            if self.nodes[i].ID != self.nodes[j].ID
                                            if Reef.distance(
                                                    self.nodes[i].location,
                                                    self.nodes[j].location)
                                            < threshold)
                      for i,val in enumerate(self.nodes)}
        self.count_neighbor_types()

    def inform(self, initial, final, nodeID):
        for n in self.graph[nodeID]:
            if n not in self.updates:
                self.updates[n] = np.zeros(3)
            self.updates[n][initial] -= 1
            self.updates[n][final] += 1

    def update(self):
        for key in self.updates:
            self.nodes[key].neighbors += self.updates[key]
        self.updates = {}

    def roll(self, r, d, a, g, y, dt):

        for i, val in enumerate(self.nodes):
            U = np.random.uniform(0,1)
            total_neighborhood_count = self.nodes[i].neighbors.sum()
            coral_density = self.nodes[i].neighbors[0]/total_neighborhood_count
            turf_density = self.nodes[i].neighbors[1]/total_neighborhood_count
            algae_density = self.nodes[i].neighbors[2]/total_neighborhood_count

            if self.nodes[i].type == 0:

                if U <  (d / (1+coral_density)) * dt:

                    self.nodes[i].type = 1
                    self.inform(initial = 0, final = 1, nodeID = i)

                elif U < (a * algae_density +
                          d / (1+coral_density)) * dt:

                    self.nodes[i].type = 2
                    self.inform(initial = 0, final = 2, nodeID = i)

            elif self.nodes[i].type == 1:

                if U < (r * coral_density) * dt:

                    self.nodes[i].type = 0
                    self.inform(initial = 1, final = 0, nodeID = i)

                elif U < (y * algae_density +
                          r * coral_density) * dt:

                    self.nodes[i].type = 2
                    self.inform(initial = 1, final = 2, nodeID = i)

            elif self.nodes[i].type == 2:

                if U < g /(1 + algae_density + turf_density) * dt:
                    self.nodes[i].type = 1
                    self.inform(initial = 2, final = 1, nodeID = i)

        self.update()

class Ocean():

    # To capture reefs going through multiple simulations

    def __init__(self):
        self.simulation = []

    def append(self, simulation):
        self.simulation.append(simulation)
