#!/bin/bash

# Define your models
models=( "unet" "unetplusplus" "manet" "linknet" "fpn" "pspnet" "pan" "deeplabv3" "deeplabv3plus" "fcbformer" "enformer_lite_mini" "enformer_lite_small" "enformer_lite_medium" "enformer_lite_large" "enformer" )

# Loop through each model
for model in "${models[@]}"; do
    
    python3.7 predict.py \
        --data-dir /path/to/your/dataset \
        --criterion bcedice \
        --in-chans 3 \
        --log-interval 100 \
        --model "$model" \
        --img-size 352 \
        --drop-rate 0.0 \
        --num-workers 0 \
        --batch-size 1 \
        --num-classes 1 \
        --pin-memory \
        --rank 7 \
        
done