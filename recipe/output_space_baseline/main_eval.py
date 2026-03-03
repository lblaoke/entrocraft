"""
Evaluate answer likelihood and output-space baseline across model checkpoints.

Follows the same distributed architecture as verl/trainer/main_ppo.py:
  - Ray workers with FSDP actor + vLLM rollout
  - Base model loaded once via init_model()
  - Checkpoints loaded via load_checkpoint() (each FSDP rank loads its own shard)
  - Responses generated via generate_sequences() (vLLM)
  - Metrics computed via compute_osb_metrics() on the actor model

Metrics are computed on model-generated responses, not ground-truth answers.

Per-sample metrics on the generated response tokens:
  a. Log-likelihood:  sum_t log p(y_t | prompt, y_{<t})
  b. Output-space baseline (log):
       sum_t  sum_{i != y_t}  (-delta_other / delta_correct) * log p_{t,i}
"""

import argparse
import glob
import json
import os
import socket
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_worker_config(args) -> OmegaConf:
    """Build the OmegaConf config expected by ActorRolloutRefWorker.

    Mirrors the Hydra-composed config from main_ppo.py but only includes
    the fields needed for eval (no optimizer, no critic, no ref policy).
    """
    cfg = OmegaConf.create({
        "hybrid_engine": True,
        "nccl_timeout": 600,
        "model": {
            "path": args.base_model,
            "trust_remote_code": True,
            "use_remove_padding": True,
            "use_shm": False,
            "use_fused_kernels": False,
            "enable_gradient_checkpointing": False,
            "enable_activation_offload": False,
            "use_liger": False,
            "lora_rank": 0,
            "override_config": {},
            "external_lib": None,
            "hf_config_path": None,
            "tokenizer_path": None,
            "custom_chat_template": None,
        },
        "actor": {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "strategy": args.actor_strategy,
            "grad_clip": 1.0,
            "ulysses_sequence_parallel_size": 1,
            "entropy_from_logits_with_chunking": False,
            "entropy_checkpointing": False,
            "use_remove_padding": True,
            "use_dynamic_bsz": False,
            "use_torch_compile": True,
            "ppo_mini_batch_size": args.batch_size * args.n_gpus_per_node,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": args.batch_size,
            "ppo_max_token_len_per_gpu": 16384,
            "use_fused_kernels": False,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_loss_type": "kl",
            "entropy_coeff": 0.0,
            "ppo_epochs": 1,
            "shuffle": False,
            "loss_agg_mode": "token-mean",
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 0.2,
            "policy_loss": {"_target_": "verl.workers.config.PolicyLossConfig"},
            "fsdp_config": {
                "_target_": "verl.workers.config.FSDPEngineConfig",
                "fsdp_size": -1,
                "param_offload": False,
                "optimizer_offload": False,
                "offload_policy": False,
                "reshard_after_forward": True,
                "forward_prefetch": True,
                "model_dtype": "fp32",
                "use_orig_params": False,
                "wrap_policy": {},
                "mixed_precision": None,
            },
            "optim": None,
            "checkpoint": {
                "_target_": "verl.trainer.config.CheckpointConfig",
                "load_contents": ["model"],
                "save_contents": [],
            },
            "profiler": {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "all_ranks": False,
                "ranks": [],
                "save_path": None,
                "tool_config": None,
            },
        },
        "rollout": {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": args.rollout_name,
            "mode": "sync",
            "temperature": args.temperature,
            "top_k": -1,
            "top_p": args.top_p,
            "prompt_length": args.max_prompt_length,
            "response_length": args.max_new_tokens,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "ignore_eos": False,
            "enforce_eager": False,
            "cudagraph_capture_sizes": None,
            "free_cache_engine": True,
            "tensor_model_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": 8192,
            "max_model_len": None,
            "max_num_seqs": 1024,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "load_format": "dummy",
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": args.batch_size,
            "log_prob_use_dynamic_bsz": False,
            "log_prob_max_token_len_per_gpu": 16384,
            "disable_log_stats": True,
            "do_sample": args.do_sample,
            "n": 1,
            "over_sample_rate": 0,
            "multi_stage_wake_up": False,
            "engine_kwargs": {"vllm": {}, "sglang": {}},
            "val_kwargs": {
                "_target_": "verl.workers.config.SamplingConfig",
                "temperature": args.temperature,
                "top_k": -1,
                "top_p": args.top_p,
                "n": 1,
                "do_sample": args.do_sample,
            },
            "multi_turn": {
                "_target_": "verl.workers.config.MultiTurnConfig",
                "enable": False,
                "max_assistant_turns": None,
                "tool_config_path": None,
                "max_user_turns": None,
                "max_parallel_calls": 1,
                "max_tool_response_length": 256,
                "tool_response_truncate_side": "middle",
                "interaction_config_path": None,
                "use_inference_chat_template": False,
                "tokenization_sanity_check_mode": "strict",
                "format": "hermes",
                "num_repeat_rollouts": None,
            },
            "calculate_log_probs": False,
            "agent": {
                "_target_": "verl.workers.config.AgentLoopConfig",
                "num_workers": 8,
                "agent_loop_config_path": None,
                "custom_async_server": {
                    "_target_": "verl.workers.config.CustomAsyncServerConfig",
                    "path": None,
                    "name": None,
                },
            },
            "update_weights_bucket_megabytes": 512,
            "trace": {
                "_target_": "verl.workers.config.TraceConfig",
                "backend": None,
                "token2text": False,
            },
            "skip_rollout": False,
            "skip_dump_dir": "/tmp/rollout_dump",
            "skip_tokenizer_init": True,
            "length_filter_keep_fraction": 1.0,
            "r_var_filter_keep_fraction": 1.0,
            "profiler": {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "all_ranks": False,
                "ranks": [],
                "save_path": None,
                "tool_config": None,
            },
            "layered_summon": False,
        },
        "ref": {
            "_target_": "verl.workers.config.FSDPRefConfig",
            "fsdp_config": {
                "_target_": "verl.workers.config.FSDPEngineConfig",
                "param_offload": True,
                "fsdp_size": -1,
            },
            "log_prob_micro_batch_size_per_gpu": args.batch_size,
            "log_prob_use_dynamic_bsz": False,
            "log_prob_max_token_len_per_gpu": 16384,
        },
    })
    return cfg


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def load_eval_data(data_path: str, tokenizer, max_prompt_length: int,
                   prompt_column: str, max_samples: int | None) -> list[dict]:
    """Load and tokenize evaluation prompts."""
    import pandas as pd

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".jsonl"):
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if max_samples is not None:
        df = df.iloc[:max_samples]

    items = []
    for _, row in df.iterrows():
        prompt = row[prompt_column]
        if isinstance(prompt, list):
            prompt_text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = str(prompt)

        prompt_ids = tokenizer(
            prompt_text, truncation=False, add_special_tokens=True
        )["input_ids"][:max_prompt_length]

        ground_truth = None
        if "reward_model" in row:
            rm = row["reward_model"]
            if isinstance(rm, dict):
                ground_truth = str(rm.get("ground_truth", ""))

        items.append({"prompt_ids": prompt_ids, "ground_truth": ground_truth})
    return items


def make_gen_dataproto(items: list[dict], tokenizer, max_prompt_length: int) -> DataProto:
    """Convert prompt items into a left-padded DataProto for generate_sequences."""
    pad_id = tokenizer.pad_token_id
    max_len = max(len(it["prompt_ids"]) for it in items)
    max_len = min(max_len, max_prompt_length)

    input_ids_lst, attn_mask_lst = [], []
    for it in items:
        ids = it["prompt_ids"][-max_len:]
        pad_len = max_len - len(ids)
        input_ids_lst.append([pad_id] * pad_len + ids)
        attn_mask_lst.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(input_ids_lst, dtype=torch.long)
    attention_mask = torch.tensor(attn_mask_lst, dtype=torch.long)
    position_ids = compute_position_id_with_mask(attention_mask)

    return DataProto.from_dict(tensors={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })


# ---------------------------------------------------------------------------
# Ray remote eval runner
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class EvalRunner:
    """Ray remote actor that mirrors main_ppo.py's TaskRunner for eval."""

    def run(self, config, eval_args):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"EvalRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        worker_config = OmegaConf.create(config)
        eval_args = OmegaConf.create(eval_args)

        # --- tokenizer (same as main_ppo.py lines 274-277) ---
        local_path = copy_to_local(
            worker_config.model.path,
            use_shm=worker_config.model.get("use_shm", False),
        )
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # --- worker setup (same as main_ppo.py lines 110-141 + ray_trainer init_workers) ---
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        mapping = {Role.ActorRollout: "global_pool"}

        resource_pool_spec = {
            "global_pool": [eval_args.n_gpus_per_node] * eval_args.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )
        resource_pool_manager.create_resource_pool()

        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        class_dict = {
            "actor_rollout": RayClassWithInitArgs(
                cls=role_worker_mapping[Role.ActorRollout],
                config=worker_config,
                role="actor_rollout",
            ),
        }

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name="cuda",
        )
        spawn_wg = wg.spawn(prefix_set=class_dict.keys())
        actor_rollout_wg = spawn_wg["actor_rollout"]
        actor_rollout_wg.init_model()

        print("Workers initialized, model loaded.")

        # --- load eval data ---
        data_paths = [p.strip() for p in eval_args.data_paths.split(",")]
        os.makedirs(eval_args.output_dir, exist_ok=True)

        # --- discover checkpoints ---
        checkpoint_root = eval_args.checkpoint_root
        ckpt_dirs = sorted(glob.glob(os.path.join(checkpoint_root, "global_step_*", "actor")))
        ckpt_labels = [Path(d).parent.name for d in ckpt_dirs]

        eval_targets = [("base", None)] + list(zip(ckpt_labels, ckpt_dirs))
        print(f"\nWill evaluate {len(eval_targets)} models: "
              f"base + {len(ckpt_dirs)} checkpoints")

        all_results = {}
        for label, ckpt_path in eval_targets:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {label}")
            print(f"{'=' * 60}")

            if ckpt_path is not None:
                actor_rollout_wg.load_checkpoint(ckpt_path)
                print(f"  Loaded checkpoint: {ckpt_path}")

            model_results = {"label": label, "checkpoint": ckpt_path, "datasets": {}}
            for data_path in data_paths:
                ds_name = Path(data_path).stem
                print(f"\n  Dataset: {ds_name}  ({data_path})")

                items = load_eval_data(
                    data_path, tokenizer,
                    max_prompt_length=eval_args.max_prompt_length,
                    prompt_column=eval_args.prompt_column,
                    max_samples=eval_args.max_samples,
                )
                if not items:
                    print("    (empty dataset, skipping)")
                    continue

                # -- generate responses --
                gen_batch = make_gen_dataproto(items, tokenizer, eval_args.max_prompt_length)
                gen_batch.meta_info = {
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": eval_args.do_sample,
                    "validate": True,
                    "global_steps": 0,
                }

                dp_size = actor_rollout_wg.world_size
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, dp_size)
                gen_output_padded = actor_rollout_wg.generate_sequences(gen_batch_padded)
                gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)

                response_ids = gen_output.batch["responses"]
                response_texts = [
                    tokenizer.decode(ids[ids != tokenizer.pad_token_id], skip_special_tokens=True)
                    for ids in response_ids
                ]
                response_lengths = (response_ids != tokenizer.pad_token_id).sum(dim=-1).tolist()

                print(f"    Generated {len(response_texts)} responses, "
                      f"mean length {np.mean(response_lengths):.1f} tokens")

                # -- compute OSB metrics via actor forward pass --
                full_batch = gen_output
                full_batch.meta_info = {
                    "micro_batch_size": eval_args.batch_size,
                    "temperature": eval_args.temperature,
                }

                full_batch_padded, pad_size2 = pad_dataproto_to_divisor(full_batch, dp_size)
                osb_output_padded = actor_rollout_wg.compute_osb_metrics(full_batch_padded)
                osb_output = unpad_dataproto(osb_output_padded, pad_size=pad_size2)

                target_lp = osb_output.batch["target_log_probs"]
                sum_lp = osb_output.batch["sum_log_probs"]

                response_mask = (response_ids != tokenizer.pad_token_id).float()
                exponent = -eval_args.delta_other / eval_args.delta_correct

                per_sample_ll = (target_lp * response_mask).sum(dim=-1).tolist()
                per_sample_osb = (exponent * (sum_lp - target_lp) * response_mask).sum(dim=-1).tolist()

                valid_ll = [v for v in per_sample_ll if not np.isnan(v)]
                valid_osb = [v for v in per_sample_osb if not np.isnan(v)]
                mean_resp_len = float(np.mean(response_lengths)) if response_lengths else 1.0

                ds_result = {
                    "num_samples": len(items),
                    "num_valid": len(valid_ll),
                    "mean_response_length": mean_resp_len,
                    "mean_log_likelihood": float(np.mean(valid_ll)) if valid_ll else float("nan"),
                    "mean_log_osb": float(np.mean(valid_osb)) if valid_osb else float("nan"),
                    "mean_log_likelihood_per_token": (
                        float(np.mean(valid_ll)) / max(mean_resp_len, 1) if valid_ll else float("nan")
                    ),
                    "mean_log_osb_per_token": (
                        float(np.mean(valid_osb)) / max(mean_resp_len, 1) if valid_osb else float("nan")
                    ),
                    "per_sample": [
                        {
                            "log_likelihood": ll,
                            "log_osb": osb,
                            "response_length": rlen,
                            "generated_response": rtxt,
                            "ground_truth": it["ground_truth"],
                        }
                        for ll, osb, rlen, rtxt, it in zip(
                            per_sample_ll, per_sample_osb, response_lengths, response_texts, items
                        )
                    ],
                }
                model_results["datasets"][ds_name] = ds_result

                print(f"    n={ds_result['num_samples']}  "
                      f"valid={ds_result['num_valid']}  "
                      f"mean_resp_len={mean_resp_len:.1f}  "
                      f"log_ll={ds_result['mean_log_likelihood']:.4f}  "
                      f"log_osb={ds_result['mean_log_osb']:.4f}  "
                      f"log_ll/tok={ds_result['mean_log_likelihood_per_token']:.4f}  "
                      f"log_osb/tok={ds_result['mean_log_osb_per_token']:.4f}")

            all_results[label] = model_results

            per_model_file = os.path.join(eval_args.output_dir, f"{label}.json")
            with open(per_model_file, "w") as f:
                json.dump(model_results, f, indent=2)
            print(f"  -> saved {per_model_file}")

        # -- compact summary --
        summary = {}
        for name, res in all_results.items():
            summary[name] = {
                "label": res["label"],
                "checkpoint": res["checkpoint"],
                "datasets": {
                    ds: {k: v for k, v in info.items() if k != "per_sample"}
                    for ds, info in res["datasets"].items()
                },
            }
        summary_file = os.path.join(eval_args.output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary -> {summary_file}")

        return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer likelihood and output-space baseline "
                    "on model-generated responses (distributed, following main_ppo.py)"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="HuggingFace Hub ID or local path to the base model",
    )
    parser.add_argument(
        "--checkpoint_root", type=str, required=True,
        help="Root dir containing global_step_*/actor checkpoints",
    )
    parser.add_argument(
        "--data_paths", type=str, required=True,
        help="Comma-separated paths to evaluation data (parquet / jsonl)",
    )
    parser.add_argument("--output_dir", type=str,
                        default="./results/output_space_baseline")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--do_sample", action=argparse.BooleanOptionalAction, default=True,
    )

    parser.add_argument("--delta_correct", type=float, default=1e-2)
    parser.add_argument("--delta_other", type=float, default=-1e-6)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )

    parser.add_argument("--n_gpus_per_node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--actor_strategy", type=str, default="fsdp2",
                        choices=["fsdp", "fsdp2"])
    parser.add_argument("--rollout_name", type=str, default="vllm",
                        choices=["vllm", "sglang", "hf"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)

    args = parser.parse_args()

    # --- build worker config ---
    worker_config = build_worker_config(args)

    # --- serialisable eval args ---
    eval_args_dict = {
        "data_paths": args.data_paths,
        "output_dir": args.output_dir,
        "prompt_column": args.prompt_column,
        "max_prompt_length": args.max_prompt_length,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "delta_correct": args.delta_correct,
        "delta_other": args.delta_other,
        "checkpoint_root": args.checkpoint_root,
        "n_gpus_per_node": args.n_gpus_per_node,
        "nnodes": args.nnodes,
    }

    # --- Ray init (same as main_ppo.py lines 55-66) ---
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
            },
        }
        ray.init(runtime_env=runtime_env)

    runner = EvalRunner.remote()
    summary = ray.get(runner.run.remote(
        OmegaConf.to_container(worker_config, resolve=True),
        eval_args_dict,
    ))

    print("\nDone. Summary:")
    for name, info in summary.items():
        for ds, metrics in info["datasets"].items():
            print(f"  {name} / {ds}: "
                  f"log_ll={metrics['mean_log_likelihood']:.4f}  "
                  f"log_osb={metrics['mean_log_osb']:.4f}")


if __name__ == "__main__":
    main()
