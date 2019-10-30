import numpy as np
import random
import math


class Organism():

    def __init__(self, ID, type, location):
        self.ID = ID
        self.type = type
        self.density = np.zeros(3)
        self.location = location

class Reef():
        # append(): Add organisms to create a Reef

        # generateGraph(): Creates a graph of appended nodes of class Organism,
        # based on their defined 2-d coordinate location

        # roll(): one stochastic update of each node, weighted odds based on
        # differential equation in documentation

    def __init__(self):
        self.nodes = []
        self.graph = {}
        self.updates = {}
        self.rolls = 0
        self.coralNodeCount = {}
        self.coralNeighborCount = {}
        #self.configs = {}

    def append(self, node):
        self.nodes.append(node)

    def distance(a,b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx*dx + dy*dy)

    def initiateDensity(self):
        for i,val in enumerate(self.nodes):
            for n in self.graph[i]:
                if self.nodes[n].type == 0:
                    self.nodes[i].density[0] += 1
                elif self.nodes[n].type == 1:
                    self.nodes[i].density[1] += 1
                elif self.nodes[n].type == 2:
                    self.nodes[i].density[2] += 1

    def generateGraph(self, threshold=1.5):
        self.graph = {self.nodes[i].ID:list(self.nodes[j].ID
                      #can add things, e.g. self.nodes[j].type
                                            for j,val in enumerate(self.nodes)
                                            if self.nodes[i].ID != self.nodes[j].ID
                                            if Reef.distance(
                                                    self.nodes[i].location,
                                                    self.nodes[j].location)
                                            < threshold)
                      for i,val in enumerate(self.nodes)}
        self.initiateDensity()

    def inform(self, initial, final, nodeID):
        for n in self.graph[nodeID]:
            if n not in self.updates:
                self.updates[n] = np.zeros(3)
            self.updates[n][initial] -= 1
            self.updates[n][final] += 1

    def update(self):
        for key in self.updates:
            self.nodes[key].density += self.updates[key]
        self.updates = {}

    def roll(self, r, d, a, g, y, dt):

        for i, val in enumerate(self.nodes):

            U = random.uniform(0,1)
            totalDensity = self.nodes[i].density.sum()
            coralDensity = self.nodes[i].density[0]/totalDensity
            turfDensity = self.nodes[i].density[1]/totalDensity
            algaeDensity = self.nodes[i].density[2]/totalDensity

            if self.nodes[i].type == 0:

                ## Counting Number of Coral Nodes and adding their Neighborhood Type Count
                ## Can divide denisties by number of coral nodes to get average
                if self.rolls not in self.coralNodeCount:
                    self.coralNodeCount[self.rolls] = 0
                    self.coralNeighborCount[self.rolls] = 0
                self.coralNodeCount[self.rolls] += 1
                self.coralNeighborCount[self.rolls] += self.nodes[i].density[0]
                ##

                if U <  (d / (1+coralDensity)) * dt:

                    self.nodes[i].type = 1
                    self.inform(initial = 0, final = 1, nodeID = i)

                elif U < (a * algaeDensity +
                          d / (1+coralDensity)) * dt:

                    self.nodes[i].type = 2
                    self.inform(initial = 0, final = 2, nodeID = i)

            elif self.nodes[i].type == 1:

                if U < (r * coralDensity) * dt:

                    self.nodes[i].type = 0
                    self.inform(initial = 1, final = 0, nodeID = i)

                elif U < (y * algaeDensity +
                          r * coralDensity) * dt:

                    self.nodes[i].type = 2
                    self.inform(initial = 1, final = 2, nodeID = i)

            elif self.nodes[i].type == 2:

                if U < g /(algaeDensity + turfDensity) * dt: #needed to add the one now that the                                                       node is not counting itself
                    self.nodes[i].type = 1
                    self.inform(initial = 2, final = 1, nodeID = i)

        self.update()
        self.rolls += 1

class Ocean():

    # To capture reefs going through multiple simulations

    def __init__(self):
        self.simulation = []

    def append(self, simulation):
        self.simulation.append(simulation)
