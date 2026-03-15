#!/bin/bash

sinteractive \
    --job-name=debug \
    --partition=ai \
    --account=ruqiz \
    --qos=normal \
    --nodes=1 \
    --ntasks-per-node=4 \
    --gres=gpu:4 \
    --cpus-per-task=28 \
    --time=0-01:00:00
