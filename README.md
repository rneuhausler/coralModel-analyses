
# coralModel

----
For the purpose of drawing attention to specific parts of the python code throughout this introduction, I substitute non-focused on parts of the code with:
```python
     .
     .
```
---- 
             
## Model Overview

coralModel is a stochastic spatiotemporal model representing the spatiotemporal evolution of three competing coral reef benthic coverages:

* Coral
* Algal turf
* Macroalgae

The model consists of various nodes, each of which is assigned one of the benthic coverage types listed above. 
Over time, a node's type updates stochastically through probabilities weighted by:
1. Overall reef conditions defined through input parameters, and
2. The node's immediate neighbors' types.

Below is an example of an 15x15 node reef's composition initially and after 100 runs (updates) 
(0=Coral, 1=Turf, 2=Macroalgae):


![](images/exampleOutput/InitialFinal.png)


### Model Structure

  We derived the rules for our model's dynamics from Mumby et al. (2007)'s reef competition ODE's, shown below [1]:

![](images/mumbyEquations.png)

From the equations above, we extract a set of 5 reactions that describe the probabilities of switching between the respective node types:

![](images/mumbyAdjusted.png)

We are consistent with Mumby et al. in considering the parameters `r`, `d`, `a`, `g`, and `y`, to represent overall reef conditions, but deviate through our use of neighborhood densities and agent-based approach. In our reactions above, we calculate `M`, `T`, `C` as local densities (based on neighborhood benthic compositions) instead of global percentages (reef-wide). We implement this through the spatial explicitness of our model.

Our model is a product of object oriented programming; we abstract benthic coverages as instances of the class `Organism() `, and appending them to an instance of the class `Reef()`. 

We define these classes in `coralModel.py` as follows:

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


As can be seen above, each `Organism` carries the knowledge of:

1. It's specific type: 0 for Coral, 1 for Turf, 2 for Macroalgae .
2. The neighborhood density: an array of length three, containing the number of each benthic type represented in the set of neighboring nodes (indexed by the type value from 1.).
3. Location: a 2 dimensional coordinate location of that node, used in determining neighboring nodes.


We define class `Reef()` to have a graph listing each appended `Organism()`'s neighbors. This is first generated when the `Reef()` is fully appended with `Organism()`s, using our function `generateGraph()`, calls for a threshold, i.e the radius distance to use when considering neighbors.

The purpose in all of this is to be able to run the function `roll()`. With this function, we check to update each of the appended `Organism()`'s type in the given `Reef()`. An `Organism()`'s type is only updated when a freshly randomly generated value (RGV) falls within the range set by the current node's conditions (calculated using the weights shown on the arrows of our reactions above).

This GitHub repository contains various python and shell scripts that allow for the user to create a reef, and take it through a set number of time steps using `roll()`. The potential initial reef setups and model variabilities are described next.   


### Model Setup

In this repository, within `/scripts`, you will find a shell script called `coralModel.sh`. Within this file you will the option to adjust the following (comments not included in the file):

* The number of cores used to run model simulations in parallel on:
```python
nProcessors=4
NumberOfSimulations=10
```
* The initial grid setup:
```python
coralPercent=33 ## percentage of initial nodes that are coral
algaePercent=33 ## percentage of initial nodes that are macroalgae
gridOption=0 ## grid options 0=random,1=checkered,2=with blob of one type in center
blobValue=0 ## only used is gridOption=2
``` 
The resultant initial grids from `gridOption` options are shown below:

![](images/exampleOutput/initialGridOptions.png)

* Grid size and radius of a nodes neighborhood:
```python
rows=15 
columns=15
threshold=1.45
```

* Time settings:
```python
recordRate=90 #frequency of recording output data
dt=.1 
tf=50 #time final
```

* Model Parameters
```python
r=1.0
d=.4 
a=.2
g=.4 #array
y=.75
```


### Model Run

Using the values set above, `coralModel.sh` calls `coralModelTest.py`.

Within `coralModelTest.py`, you will find `runModel()` as well as some of the other functions defined in this one:

```python
def runModel(simulation):
        
    Moorea = createReef()
    Moorea.generateGraph(threshold) 
    
    for timestep in range(0,NumberOfTimesteps):
        if timestep == 0:
            table = pd.DataFrame([])
        if timestep % recordRate == 0:
            table = pd.concat([table, pullInfo(Moorea, simulation, timestep)])
            
        Moorea.roll(r=r, d=d, a=a, g=g, y=y, dt=dt)
        
    return(table)
    .
    .
```

As mention, `roll()` updates each node (i.e. instance of class `Organism()` within class `Reef()`) based a probability weighted by neighboring benthic coverages, determined by `generateGraph()`, and overall reef conditions, and a randomly generated number. If the randomly generated number falls within the bounds of the weighted probability, the node switches to a different type. 
These reactions emphasize the influence the parameters, pulled from the mumby equations, and species density around each node have on changing a spot on the reef from being one type to the other (e.g. the first reaction describes a coral (node) becoming macroalgae at the growth rate of macroalgae over coral and density of the two species. Whereas the mumby et al. equations consider the *global* population of coral of the system, we focus on the *local* composition around each node with assumptions such that if there is a macroalgae right next to the coral, that coral is more likely to switch to macroalgae than if it were surrounded by only other coral. This neighborhood information is stored within each node as the attribute `density` from class `Organism()`. This value is initially measured with `generateGraph()` and then updated throughout the simulation with `inform()` and `update()`, which are not shown in this introduction but can be found in `coralModel.py`.

The inclusion of the local type density can be seen in the code below, showing how the function `roll()` multiplies each reaction parameter with the density of specific types in the node's neighborhood in calculating the probability of type switching.

```python

    def roll(self, r, d, a, g, y, dt):
        for i, val in enumerate(self.nodes):      
            U = random.uniform(0,1)
            totalDensity = self.nodes[i].density.sum()
            coralDensity = self.nodes[i].density[0]/totalDensity
            turfDensity = self.nodes[i].density[1]/totalDensity
            algaeDensity = self.nodes[i].density[2]/totalDensity
            
            if self.nodes[i].type == 0:   
                if U <  (d * (1+coralDensity)) * dt:                    # <-- reaction parameter * density
                    self.nodes[i].type = 1
                    self.inform(initial = 0, final = 1, nodeID = i)
                elif U < (a * (1+algaeDensity) * (1+turfDensity) + 
                          d * (1+coralDensity)) * dt:
                    self.nodes[i].type = 2
                    self.inform(initial = 0, final = 2, nodeID = i)

            elif self.nodes[i].type == 1:
                if U > 1 - (r * (1+coralDensity) * (1+turfDensity)) * dt:
                    self.nodes[i].type = 0
                    self.inform(initial = 1, final = 0, nodeID = i)
    .
    .
            elif self.nodes[i].type == 2:
    .
    .
```


## Outputs

We pull and save the following metrics, shown in the example output below:

![](images/exampleOutput/table.png)

(Coral-CoralNeighbors represents the average number of coral neighbors for a coral node (same holds for Turf-TurfNeighbors and Macro-MacroNeighbors).) 


Currently, `coralModel.sh` creates a folder network to organized model outputs based on the inputs, as seen below:

`output/15x15/grid0/grazing30/coral33-macro33-r10-d40-a20-y75-time5010-rec500-nsim100.csv`

(i.e. output, rows x columns, initial grid option, grazing value (x100 to remove decimals), 
coral percent, macroalgae percent, inputs r, d, a, y, final time, record rate, and number of simulations)

You can visualize the output data using `modelOutputViewer.ipynb`. 


## Running coralModelTest.py and Exploring Outputs

To run the `coralModelTest.py`, follow the following instructions:

1. Make sure you have python 3.6 installed
2. Open your terminal and `cd` to the location where you wish to store this repository
e.g.
```
cd Documents/models
```
3. clone the repository and `cd` into the scripts file
```
git clone https://github.com/rneuhausler/coralModel
cd coralModel/scripts
```
4. Run the model
```
sh coralModel-grazingLoop.sh
```

Once the model is done running, you should see a folder titled `/output` in the `/scripts` folder. In here, you will find all the csv files (described above in Outputs and Metrics) organized into folders (mentioned above in Outputs and Metrics). 

To explore your outputs:

1. Open jupyter notebook (or lab)
```
jupyter notebook 
```
2. Within jupyter, open `modelOutputViewer.ipynb`



## References

[1] Mumby, P. J., Hastings, A., & Edwards, H. J. (2007). Thresholds and the resilience of Caribbean coral reefs. Nature, 450(7166), 98–101. https://doi.org/10.1038/nature06252





