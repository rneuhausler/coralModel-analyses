Spatiotemporal Coral Model Overview
================
Rosanna Neuhausler |
June 14th, 2019

## Issues with running the model

Error message when compiling in virtual environment creaded via

``` r
conda create -n coral3 python=x.x anaconda
```

The message recieved:

``` r
error: use of undeclared identifier 'PyString_AsString'
    char* str = PyString_AsString(thisAttr);
```

Which leads to:

``` r
Traceback (most recent call last):
  File "nofish.py", line 1, in <module>
    import sparpy
ImportError: No module named sparpy
```

## Abstract W/o Fish

Ecological phase shifts, from coral reefs to macroalgae dominated
states, have drastically changed tropical coastal zones over the past 60
years. However, these shifts are not always immediate - macroalgae are
generally present even when coral colonies are dominant, with algal
distribution regulated by herbivorous fish. Current efforts in
projecting coral-algal phase shifts utilize temporal mechanistic models
and spatiotemporal statistical models, both of which fail to capture
metastability between multiple steady states given a lack of spatial
information and spatiotemporal resolution, respectively. To address
these concerns, we build on these models to account for spatial
variations and stochasticity, in combination with individual organisms
and their mechanisms. Our model can project the percent cover and
spatial distributions of coral, macroalgae, and algae turf as a function
of three reef resilience indicators - herbivorous grazers, water
quality, and neighborhing species. Our model is . Water quality and
coral demographics are input parameters that can vary over time,
allowing our model to be run for temporally changing scenarios that can
be adjusted for different reefs.

## Abstract W/ Fish

Ecological phase shifts, from coral reefs to macroalgae dominated
states, have drastically changed tropical coastal zones over the past 60
years. However, these shifts are not always immediate - macroalgae are
generally present even when coral colonies are dominant, with algal
distribution regulated by herbivorous fish. Current efforts in
projecting coral-algal phase shifts utilize temporal mechanistic models
and spatiotemporal statistical models, both of which fail to capture
metastability between multiple steady states given a lack of spatial
information and spatiotemporal resolution, respectively. To address
these concerns, we build on these models to account for spatial
variations and stochasticity, in combination with individual organisms
and their mechanisms. Our model can project the percent cover and
spatial distributions of coral, macroalgae, and algae turf as a function
of three reef resilience indicators - herbivorous grazers, water
quality, and neighborhing species. Our model is . Water quality and
coral demographics are input parameters that can vary over time,
allowing our model to be run for temporally changing scenarios that can
be adjusted for different reefs.

**Key Words** : Coral Reef Competition, Spatiotemporal Modeling, Benthic
Coverage

## Introduction

Coral Reef Systems are currently threatened from temperature rise. The
main issue is not whether coral bleach, but it is their death and
inability to grow back as a result of different organisms taking their
space. We look to capture the outcomes of spatial competition through a
spatiotemporal toy model.

## Coral Reef Competition and Alternate Stable States

Coral reefs are developed through the secretion of calcium carbonate,
primarily by species of hard coral, over thousands of years
\[citation\]. When located in shallow waters, coral reefs are considered
optimal locations for benthic organisms, such as macroalgae and algae
turf, to grow upon due to sunlight availability. Coral and algae
together, through their facilitation of housing and food, attract a
variety of other organisms, resulting in “rainforests of the ocean”
(Swart 2013). However, these systems have been observed to switch to a
highly-less diverse state when external factors drive algae to
outcompete coral for reef space. In these cases the reef is referred to
as “algae-dominated”, and they have been used as evidence for coral reef
systems as having alternate stable states, a claim under complex systems
theory. \[CITE\].

Unique to complex systems is the idea of regime shifts - a change from
one stable state to another often due to an external force pushing the
system past a specific threshold. This concept is visualized in the
figure to the right \[insert figure of curve with ball\]. While many
scientists have been… \[What does this debate mean?\] This phenomena is
often discussed in dichotomy, with the system either being in one state
or the other. However, In our study we look to explore the transition
between these states, specifically metastability and reefs inclined to
frequent switching.

Various factors are thought to change the odds for which competitor
(coral or algae) dominates the reef, through either boosting or impeding
the growth of macroalgae, or the destruction of coral. These are shown
in Figure \ref{fig:reefsystem}.

**Coral System Image here**

caption:The basic interactions of a natural coral reef system, impacting
whether coral or algae is dominating the reef, are show above. When
humans intervene this system, either through overfishing, increasing the
input of nutrients, or breaking off the coral themselves, the odds for
which benthic organism dominates changes in favor of the algaes. This
puts competitive pressure on the coral.

Coral dominated systems, with limited algae presence, are the prefered
state due to their importance for marine life and human dependency on
fishery success/sustainability as well as coral reef tourism. Within the
coral triangle alone - Malaysia, Indonesia, Papua New Guinea, and the
Philippines - the estimated economic returns from coral reefs for
tourism and fisheries were 5.85 billions and 6.23 billion USD in 2017
(UNEP, 2018).

## Current Efforts in Modeling Reef Dynamics

### Reef Competition Equations

  - \(\frac{dM}{dt} = aMC - \frac{gM}{M+T} + \gamma MT\)
  - \(\frac{dC}{dt} = rTC - dC - aMC\)
  - \(\frac{dT}{dt} = \frac{gM}{M+T} - \gamma MT - rTC + dC\)

<!-- end list -->

``` r
output <- mumby_eqs(g = .2,  dt = .1)
mumby_plot(output, title="Grazing = .2")
```

<img src="overview_files/figure-gfm/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

and as we have shown, reef competition has been seen to cause
oscillating competitive advantages, which the mumby et al. equations do
not present on their own. We hypothesize that this fluctuating behaviour
can be captured in a mathematical model if the spatial complexity,
specifically neighborhood influences, is incorporated into the model.
Inspired by current data gathering efforts, we set up our model.

## Methodology

We build on the models described above through the introduction of
stochasticity and spatial knowledge.

#### Interventions and further exploration potential of our model

  - Stochastic update based on random variable and probability
    thresholds dependent on natural switching forces

  - Different weights of features that decided probability

  - Knowledge of neighbors (potential to also have knowledge of
    neighbors stored)

  - 
We use a python front-end, C++ back-end package, Aboria, to

The model is experimented with on various levels, intending to replicate
the effects of numbers of organisms in the reef improving that organisms
resiliency.

##### Sudo code of our probability-based grid updating scheme

``` r
    for i in range(number_of_observations):
        for j in range(grid_updates_per_observation):
  
            for p in fixed:
                density = [0.0,0.0,0.0,0.0]  #no memory of past grid set-up
            simulation.integrate(grid_dt, dt)
            
            for p in fixed:
                U = random.uniform(0,1) #randomly generated for each node
            
                if p.species == Coral:
                    if U < d * grid_dt:
                        p.species = Turf
                    elif U < d * grid_dt + a * p.density[M['type']] * grid_dt: 
                        p.species = Macroalgae
                    
                if p.species == Turf:
                    if U < gamma * grid_dt * p.density[Macroalgae] :
                        p.species = Macroalgae
                    if U < (gamma * grid_dt * p.density[Macroalgae] + 
                            r * grid_dt * p.density[Coral]):
                        p.species = Coral
                    
                if p.species == Macroalgae:
                    if U < (gu + gp * p.density[Fish]) * grid_dt:
                        p.species = Turf
```

Through the use of a probability-based grid updating scheme, we hope to
capture some of the randomness that can occur in nature, either keeping
a benthic organism from changing or not.

As can be seen above, the current scheme has knowledge of the
neighboring assemblages set to zero.

## Results

We ran our model at various parameter set ups for 100 times each, the
results from some of these run can be seen in Figure *.* below. As

## Discussion

Incorporating neighborhood knowledge into our model has allowed for

## Bibliography
