
for GRAZING in .5 .6
do
    for GRID in 0 2
    do
        for INITIALCORAL in 5 10 20
        do
            for INITIALMACRO in 5 10 20
            do
            # Inputs
                nProcessors=4
                NumberOfSimulations=100

                coralPercent=$INITIALCORAL
                algaePercent=$INITIALMACRO
                gridOption=$GRID #array

                #Time and Grid Settings
                rows=15 
                columns=15
                threshold=1.45 

                recordRate=200

            #loop through g
                r=1.0
                d=.4 
                a=.2
                g=$GRAZING
                y=.75
                dt=.1 
                tf=501 #can play with this value as well

                blobValue=0

            ## create directory
                grazingFolder=$(python -c "print(int($g*100))")
                mkdir -p 'output'/$rows'x'$columns/'grid'$gridOption/'grazing'$grazingFolder

            ## run simulation
                python coralModelTest.py $nProcessors $NumberOfSimulations $coralPercent $algaePercent $gridOption $rows $columns $threshold $recordRate $r $d $a $g $y $dt $tf $blobValue
            done
        done
    done
done

## histograms - count, neighbors, npatches - nice to compare different initial grids
##looking at all the simulations at different timesteps (reached equilibrium if looking the same after 1000 timesteps)
## play with final time
## output at t // 1000 (tmod 1000)
