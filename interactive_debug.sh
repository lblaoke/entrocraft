#!/bin/bash

sinteractive \
    --job-name=debug \
    --partition=ai \
    --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
    --qos=preemptible \
    --nodes=1 \
    --ntasks-per-node=8 \
    --gres=gpu:8 \
    --cpus-per-task=14 \
    --time=0-00:30:00
