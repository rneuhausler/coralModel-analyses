for VARIABLE in .5 .55
do
    for VARIABLEgrid in 2
    do
        for VARIABLEsize in 15 20
        do

            # Inputs
            nProcessors=4
            NumberOfSimulations=100

            coralPercent=33
            algaePercent=33
            gridOption=$VARIABLEgrid #array

            #Time and Grid Settings
            rows=$VARIABLEsize 
            columns=$VARIABLEsize
            threshold=1.45 

            recordRate=500

        #loop through g
            r=1.0
            d=.4 
            a=.2
            g=$VARIABLE
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


## histograms - count, neighbors, npatches - nice to compare different initial grids
##looking at all the simulations at different timesteps (reached equilibrium if looking the same after 1000 timesteps)
## play with final time
## output at t // 1000 (tmod 1000)
