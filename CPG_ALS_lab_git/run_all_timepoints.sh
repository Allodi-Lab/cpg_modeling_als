#!/bin/bash
# Run tests with predefined test names for each iteration

# Generate random numbers in Python using [np.random.randint(10**7) for i in range(25)]
seed_array=(6780264 3918556 9881035 2173291 3141774 8785002 7145648 637999 4597512 1350144 435612 2803573 3757229 6679153 8977913 3814528 4666589 8476668 8201921 2835993 506239 923180 1027114 5713566 1487307)
#seed_array=(435612)

# Define the test names array
#test_names=("P0_D1" "P45_D1" "P63_D1" "P112_D1") #run all
test_names=("P45_D1" "P63_D1" "P112_D1")   #run only disease timepoints
#test_names=("P63_D1" "P112_D1")

# Run through each test using the seed_array
for i in "${!test_names[@]}"; do
    test_name="${test_names[i]}"
    echo "Running test iteration $test_name"
    counter=1
    for seed in "${seed_array[@]}"; do
        python3 create_cpg.py "$seed" "$test_name"
        echo "Seed $seed (Iteration $counter of test $test_name)"
        ((counter++))
    done
    echo "Test iteration $test_name complete."
done

echo "Done."
