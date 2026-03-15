#!/bin/bash

sbatch scripts/grpo/grpo_range_linear/qwen3-4b-base_numina_math_full_grpo_range_linear_n8.slurm

sbatch scripts/grpo/qwen3-4b-base_numina_math_grpo_n8.slurm
sbatch scripts/grpo/grpo_entropy_loss/qwen3-4b-base_numina_math_grpo_entropy_loss_n8.slurm
sbatch scripts/grpo/grpo_clip_higher/qwen3-4b-base_numina_math_grpo_clip_higher_n8.slurm
sbatch scripts/w_reinforce/qwen3-4b-base_numina_math_w_reinforce_n8.slurm
sbatch scripts/entropic/qwen3-4b-base_numina_math_entropic_0.5_n8.slurm
