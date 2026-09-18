#!/bin/bash

SCRIPT="solver.py"

seed=1

for N in $(seq 15 25); do
  for M in $(seq 10 10 100); do

    for rep in $(seq 1 10); do

      FILE="n${N}m${M}i${seed}.json"

      if [ -f "$FILE" ]; then
        echo "Running $FILE"
        python3 $SCRIPT "$FILE"
      else
        echo "Missing $FILE, skipping"
      fi

      seed=$((seed + 1))

    done

  done
done
