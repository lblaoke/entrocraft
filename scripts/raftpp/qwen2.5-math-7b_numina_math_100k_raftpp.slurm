#!/bin/bash

#SBATCH --job-name=qwen2.5-math-7b_numina_math_100k_raftpp
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --time=0-8:00:00
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li4468@purdue.edu

# Load environment modules
module load cuda/12.6.0
module load conda/2024.09
conda activate verl

data_root="./data"
train_files="['$data_root/numina_math_100k/train.parquet']"

eval_math500=$data_root/math500/test.parquet
eval_amc23=$data_root/amc23/test.parquet
eval_aime24=$data_root/aime24/test.parquet
eval_aime25=$data_root/aime25/test.parquet

test_files="['$eval_math500', '$eval_amc23', '$eval_aime24', '$eval_aime25']"

model_name="Qwen/Qwen2.5-Math-7B"
rollout_n=8
batch_size_per_gpu=32

rollout_length_filter_keep_fraction=1.0
rollout_r_var_filter_keep_fraction=1.0

project_name="rl_dist_shift"
experiment_name="qwen2.5-math-7b_numina_math_100k_raftpp_rollout$rollout_n"

# Auto-detect number of GPUs and nodes
n_gpus_per_node=""
nnodes=1

# Method 1: Check if CUDA_VISIBLE_DEVICES is explicitly set
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Count GPUs from CUDA_VISIBLE_DEVICES (comma-separated list)
    n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (detected $n_gpus_per_node GPUs)"
fi

# Method 2: Use nvidia-smi as fallback
if [ -z "$n_gpus_per_node" ] && command -v nvidia-smi &> /dev/null; then
    n_gpus_per_node=$(nvidia-smi --list-gpus | wc -l)
fi

# Method 3: Default fallback
if [ -z "$n_gpus_per_node" ] || [ "$n_gpus_per_node" -eq 0 ]; then
    echo "Warning: Could not detect GPUs. Using default value of 8"
    n_gpus_per_node=8
fi

echo "Auto-detected configuration: nnodes=$nnodes, n_gpus_per_node=$n_gpus_per_node"

# Set logger based on whether running as a SLURM job or directly with bash
# If SLURM_JOB_NAME is "debug", disable wandb logging
if [ ! -z "$SLURM_JOB_ID" ] && [ ! -z "$SLURM_JOB_NAME" ] && [ "$SLURM_JOB_NAME" != "debug" ]; then
    logger='["console","wandb"]'
    echo "Running as a SLURM job (ID: $SLURM_JOB_ID, Name: $SLURM_JOB_NAME) - using wandb logging"
else
    logger='["console"]'
    echo "Running directly with bash or debug mode - using console logging only"
    [ "$SLURM_JOB_NAME" = "debug" ] && echo "Debug mode detected - wandb logging disabled"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=raftpp \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(( n_gpus_per_node * batch_size_per_gpu )) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.length_filter_keep_fraction=$rollout_length_filter_keep_fraction \
    actor_rollout_ref.rollout.r_var_filter_keep_fraction=$rollout_r_var_filter_keep_fraction \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_activation_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="$logger" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
