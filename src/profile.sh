#!/bin/bash

echo "Beginning profiling..."

# make degree list
# make graph size
Degree=(5 10 50 100 500 800) # file
Scale=(10 12 15 20) # dir
Warp=(1 4 8 16 32) # each entry in file


# 1 . generate graph
# 2.  profile graph by invoking kenrel with different values of w
# 3.  log results


for s in "${Scale[@]}"
do
    DIR="profile-scale-$s"
    mkdir -p $DIR
    echo "Making DIR = ./$DIR"
    for d in "${Degree[@]}"
    do
        LOG_FILE="./$DIR/scale-$s-degree-$d-profile.txt"
        echo "Making LOG FILE = $LOG_FILE"
        for w in "${Warp[@]}"
        do
            nvcc -lineinfo -std=c++11 graph.cpp generator.cpp aggregate.cu test_aggregate_dyn_script.cu -DSCALE=$s -DDEGREE=$d -DWARP=$w -o dyn_script
            ./dyn_script | grep "Kernel" >> $LOG_FILE
        done
    done
done
