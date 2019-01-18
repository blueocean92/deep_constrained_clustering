#!/usr/bin/env bash
##!/usr/bin/env bash

TASKS="fashion_sdae_weights.pt \
fashion_triplet_embedding.npy \
mnist_sdae_weights.pt \
mnist_triplet_embedding.npy \
reuters10k_sdae_weights.pt"


for t in $TASKS; do
    echo "Downloading model ${t}."
    wget "https://s3-us-west-1.amazonaws.com/deep-constrained-clustering/\
model-log-final/${t}" -P ./model/
done