#!/bin/bash

sinteractive \
    --job-name=debug \
    --partition=ai \
    --account=ruqiz \
    --qos=normal \
    --nodes=1 \
    --ntasks-per-node=8 \
    --gres=gpu:8 \
    --cpus-per-task=14 \
    --time=0-01:00:00
