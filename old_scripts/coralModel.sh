### Inputs
nProcessors=4
NumberOfSimulations=10

coralPercent=33
algaePercent=33
gridOption=0 #array

#Time and Grid Settings
rows=15 
columns=15
threshold=1.45 

recordRate=90
imageReturn=true

    ## table for first run. later averages
#loop through g, nested loop through gridoption (once ready)
r=1.0
d=.4 
a=.2
g=.4 #array
y=.75
dt=.1 
tf=50 #can play with this value as well

blobValue=0

## create directory
grazingFolder=$(python -c "print(int($g*100))")
mkdir -p 'output'/$rows'x'$columns/'grid'$gridOption/'grazing'$grazingFolder/'images'

## run simulation
python coralModel_functions.py $nProcessors $NumberOfSimulations $coralPercent $algaePercent $gridOption $rows $columns $threshold $recordRate $imageReturn $r $d $a $g $y $dt $tf $blobValue



## histograms - count, neighbors, npatches - nice to compare different initial grids
##looking at all the simulations at different timesteps (reached equilibrium if looking the same after 1000 timesteps)
## play with final time
## output at t // 1000 (tmod 1000)