#!/usr/bin/env bash
##!/usr/bin/env bash

TASKS="reutersidf10k_train.npy \
reutersidf10k_test.npy"


for t in $TASKS; do
    echo "Downloading model ${t}."
    wget "https://s3-us-west-1.amazonaws.com/deep-constrained-clustering/\
Data-Reuters/${t}" -P ./experiments/dataset/reuters/
done
