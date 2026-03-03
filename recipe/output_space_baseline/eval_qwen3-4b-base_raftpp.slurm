#!/bin/bash
#SBATCH --job-name=eval_output_space_baseline_raftpp
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --time=0-1:00:00
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li4468@purdue.edu

# Load environment
module load cuda/12.6.0
module load conda/2024.09
conda activate verl

# ---------------------------------------------------------------------------
# Configuration  (edit these)
# ---------------------------------------------------------------------------
BASE_MODEL="Qwen/Qwen3-4B-Base"
CHECKPOINT_ROOT="checkpoints/antifragile_rl/qwen3-4b-base_numina_math_raftpp_n8"

# Evaluation data
DATA_ROOT="./data"
DATA_PATHS="${DATA_ROOT}/math500/test.parquet"

OUTPUT_DIR="./recipe/output_space_baseline/raftpp_results"
BATCH_SIZE=8
MAX_PROMPT_LENGTH=1024
MAX_NEW_TOKENS=3072
MAX_SAMPLES=10
TEMPERATURE=0.6
TOP_P=0.95
DTYPE="bfloat16"

# Distributed settings
N_GPUS_PER_NODE=8
ACTOR_STRATEGY="fsdp2"
ROLLOUT_NAME="vllm"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.7

mkdir -p "$OUTPUT_DIR" ./logs

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
python recipe/output_space_baseline/main_eval.py \
    --base_model       "$BASE_MODEL" \
    --checkpoint_root  "$CHECKPOINT_ROOT" \
    --data_paths       "$DATA_PATHS" \
    --output_dir       "$OUTPUT_DIR" \
    --batch_size       "$BATCH_SIZE" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens   "$MAX_NEW_TOKENS" \
    --temperature      "$TEMPERATURE" \
    --top_p            "$TOP_P" \
    --do_sample \
    --dtype            "$DTYPE" \
    --max_samples      "$MAX_SAMPLES" \
    --n_gpus_per_node  "$N_GPUS_PER_NODE" \
    --actor_strategy   "$ACTOR_STRATEGY" \
    --rollout_name     "$ROLLOUT_NAME" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"

echo ""
echo "Done.  Results in ${OUTPUT_DIR}/summary.json"
