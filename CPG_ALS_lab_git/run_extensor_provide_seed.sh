#!/bin/bash
# Basic while loop

#Generate random numbers in Python using [np.random.randint(10**7) for i in range(20)]
#seed_array=(6780264 3918556 9881035 2173291 3141774 8785002 7145648 637999 4597512 1350144 435612 2803573 3757229 6679153 8977913 3814528 4666589 8476668 8201921 2835993 506239 923180 1027114 5713566 1487307)
seed_array=(3918556)
counter=1
for seed in "${seed_array[@]}"; do
    python3 isolated_extensor.py $seed 
    #echo $seed
    echo $counter
    ((counter++))
done

echo All done
