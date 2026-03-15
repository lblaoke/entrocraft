#!/bin/bash

#SBATCH --job-name=eval_math_from_ckpt_16384
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --time=0-0:30:00
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li4468@purdue.edu

# Load environment modules
module load cuda/12.6.0
module load conda/2024.09
conda activate verl

data_root="./data"

eval_math500=$data_root/math500/test.parquet
eval_amc23=$data_root/amc23/test.parquet
eval_aime24=$data_root/aime24/test.parquet
eval_aime25=$data_root/aime25/test.parquet

test_files="['$eval_math500','$eval_amc23','$eval_aime24','$eval_aime25']"

model_name="Qwen/Qwen3-4B-Base"
checkpoint_folder="checkpoints/antifragile_rl/qwen3-4b-base_numina_math_grpo_n8/global_step_101"
batch_size_per_gpu=32

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

logger='["console"]'

python3 -m verl.trainer.main_ppo \
    trainer.val_only=True \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path=$checkpoint_folder \
    data.train_files="$test_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$(( n_gpus_per_node * batch_size_per_gpu )) \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(( n_gpus_per_node * batch_size_per_gpu )) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    trainer.logger="$logger" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.total_epochs=1 $@
