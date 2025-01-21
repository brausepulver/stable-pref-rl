#!/usr/bin/env bash

seed="0,1,2,3,4,5,6,7,8,9"
python -m train preset=ppo/quadruped_walk training.seed="$seed" "$@" --multirun
