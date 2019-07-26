

# coralModel*
*official name tbd


## Overview

coralModel is a stochastic spatiotemporal model representing the spatiotemporal evolution of three competing coral reef benthic coverages:

* coral
* algal turf
* macroalgae

The model updates stochastically through probabilities weighted by neighboring benthic coverages and overall reef conditions.


### Model Structure

Benthic coverages are abstracted as instances of the class `Organism() `, which are appended to the an instance of the class `Reef()`.

```python
class Organism():  
    def __init__(self, ID, type, location):
        self.ID = ID
        self.type = type
        self.density = np.zeros(3)
        self.location = location
        
class Reef():
    def __init__(self):
        self.nodes = []
        self.graph = {}
        self.updates = {}
        
    def append(self, node):
     .
     .
    def generateGraph(self, threshold=1.5)
     .
     .
    def roll(self, r, d, a, g, y, dt):
     .
     .
```

The user establishes instances of class `Organism()` with a benthic type (0=coral, 1=turf, 2=macroalgae), a coordinate location, and an ID number. These instances can then be appended to an instance of class `Reef()` as a node attribute, using `append()`. Once all the nodes are appended, the user can use `generateGraph()` to establish which instances of class `Organism()` are considered as neighbors of oneanother (based on a given distance threshold and the previously defined coordinate location). 


Once the graph is generated, the user can run a timestep of the model through `roll()`. This function takes in the weights of the parameters shown in mumby et. al's ODE's and a dt.


`coralModelTest.py` provides an example model using these classes. 









calculated using Mumby et al. (2014)'s reef competion ODE's, described in the model section below.







This repository contains the files `coralModel.py`, and `coralModelTest.py`.

The first file defines the classes used to generate the spatiotemporal reef model. The second is an exmaple script of how to use the classes to create your own reef model.





