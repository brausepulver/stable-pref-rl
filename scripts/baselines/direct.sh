#!/usr/bin/env bash

for seed in $(seq 0 4 36); do
    echo "Running with seed ${seed}"
    train \
        --config-name direct \
        preset=direct/direct \
        training.seed=${seed} \
        'logging.tags=[debug]' \
        logging.group=debug &
done

wait
echo "All runs have completed."
