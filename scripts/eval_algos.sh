#!/bin/bash

# Usage: bash eval_algos.sh num_frames [envs] [algos] [seeds]

IFS=';' read -ra envs <<< "$2"
IFS=';' read -ra algos <<< "$3"
IFS=';' read -ra seeds <<< "$4"

for e in "${envs[@]}"; do
    for a in "${algos[@]}"; do
        model_name=${e}_${a}_${s}
        makedir evaluation/$model_name
        for s in ${seeds[@]}; do
            inner_model_name=seed_${s}
            python -m scripts.train --algo $a --env $e --model $inner_model_name --save-interval 10 --seed $s --frames $1
            mv storage/$inner_model_name evaluation/$model_name/
        done
    done
done