# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import warnings
import torch
import os
import random
import numpy as np
import json
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import ModelConfig, RewardConfig
from accelerate import Accelerator
from safetensors.torch import load_file 

try:
    from swanlab.integration.huggingface import SwanLabCallback
except ImportError:
    SwanLabCallback = None

# 引入我们刚才改写的模块
from rm_regression import (
    RewardModelWithValueHead, 
    RegressionRMTrainer, 
    RegressionDataCollator, 
    preprocess_regression_dataset,
    ComputeMetrics
)

tqdm.pandas()

def seed_torch(seed=42, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    # 解析参数
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to generated JSONL training data")
    parser.add_argument('--test_data_path', type=str, default=None)  
    parser.add_argument('--metrics_path', type=str, default=None)
    parser.add_argument('--linear_tpye', type=str, default="single")
    parser.add_argument('--attn_impl', type=str, default="sdpa")
    parser.add_argument('--initial_type', type=str, default="False")
    parser.add_argument('--iter', type=str, default='0')
    parser.add_argument('--dataset_num_proc', type=int, default=1)
    parser.add_argument('--quick_eval_samples', type=int, default=40)
    parser.add_argument('--enable_swanlab', action='store_true')
    parser.add_argument('--swanlab_project', type=str, default="Android-World-RStar")
    parser.add_argument('--deterministic', action='store_true')

    config, model_config, remain_args = parser.parse_args_into_dataclasses()
    if not remain_args.enable_swanlab and getattr(config, "report_to", None) in (None, "all", ["all"]):
        config.report_to = []

    # print("\n[DEBUG] Starting train_regression script...")
    # print(f"[DEBUG] Loading model from: {model_config.model_name_or_path}")

    # 强制覆盖一些参数以适配 RM 训练
    config.save_only_model = True
    config.load_best_model_at_end = False
    config.gradient_checkpointing = True
    config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    # config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    # 随机种子
    seed_torch(config.seed, deterministic=remain_args.deterministic)

    # Accelerator 准备
    accelerator = Accelerator()
    rank = accelerator.process_index

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=True, 
        use_fast=True,
        padding_side="right",
        split_special_tokens=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 Base Model
    rank0_print(rank, f"Loading model from {model_config.model_name_or_path}...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "use_cache": False,
    }
    if remain_args.attn_impl and remain_args.attn_impl.lower() != "auto":
        model_kwargs["attn_implementation"] = remain_args.attn_impl
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )
    
    # 特殊 Token 处理 (如果需要的话)
    special_tokens = [
        '<code>', '<end_of_step>', '<end_of_code>', '<output>', 
        '<end_of_output>', '<answer>', '<end_of_answer>', 
        '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>'
    ]
    if len(special_tokens) > 0:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens},
            replace_additional_special_tokens=False,
        )
        base_model.resize_token_embeddings(len(tokenizer))
    

    # 3. 包装成 Value Head 模型
    model = RewardModelWithValueHead(pretrained_model=base_model, linear_tpye=remain_args.linear_tpye)

    # 对 backbone 开启 gradient checkpointing
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "gradient_checkpointing_enable"):
        model.pretrained_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs)
    elif hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs)

    # 有些模型还需要显式设置 config
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "config"):
        model.pretrained_model.config.use_cache = False

    ckpt_path = model_config.model_name_or_path
    
    # 判断是否是一个包含权重的本地目录
    should_restore_rm_weights = remain_args.initial_type == "True" or (
        remain_args.initial_type == "False" and remain_args.iter != '0'
    )
    if should_restore_rm_weights:
        if os.path.isdir(ckpt_path):
            st_file = os.path.join(ckpt_path, "model.safetensors")
            bin_file = os.path.join(ckpt_path, "pytorch_model.bin")
            
            state_dict = None
            if os.path.exists(st_file):
                rank0_print(rank, f"Found safetensors, loading weights from {st_file}...")
                state_dict = load_file(st_file)
            elif os.path.exists(bin_file):
                rank0_print(rank, f"Found bin, loading weights from {bin_file}...")
                state_dict = torch.load(bin_file, map_location="cpu")
            
            if state_dict is not None:
                # 这里的 model 是你的 Wrapper (包含 .pretrained_model 和 .value_head)
                # 你的 Checkpoint 里存的 key 应该也是对应的结构
                # use strict=False 是为了防止一些细微的版本差异报错，只要 v_head 加载上就行
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                rank0_print(rank, f"Weights Restored. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                
                # --- 验证一下 Value Head 是否真的加载了 ---
                v_head_keys = [k for k in state_dict.keys() if "value_head" in k or "v_head" in k]
                if len(v_head_keys) > 0:
                    rank0_print(rank, f">>> SUCCESS: Loaded {len(v_head_keys)} keys for Value Head!")
                else:
                    rank0_print(rank, f">>> WARNING: Checkpoint loaded but NO Value Head keys found! Is this a fresh base model?")
    
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        
    # model.to(torch.bfloat16)

    # 4. 加载数据集
    # print(f"[DEBUG] Loading dataset from: {remain_args.train_data_path}")
    rank0_print(rank, f"Loading dataset from {remain_args.train_data_path}")
    raw_datasets = load_dataset('json', data_files=remain_args.train_data_path)
    
    # 划分训练/测试集
    if remain_args.test_data_path is not None:
        raw_datasets['test'] = load_dataset('json', data_files=remain_args.test_data_path)['train']
    else:
        # 手动划分 5%
        split = raw_datasets['train'].train_test_split(test_size=0.05, seed=42)
        raw_datasets['train'] = split['train']
        raw_datasets['test'] = split['test']
        
    rank0_print(rank, 'Train size:', len(raw_datasets['train']), 'Test size:', len(raw_datasets['test']))

    # 5. 数据预处理
    # 移除原有的列 (text, label, metadata等)
    column_names = raw_datasets['train'].column_names
    
    preprocess_func = partial(preprocess_regression_dataset, tokenizer=tokenizer, max_length=config.max_length)
    
    processed_datasets = raw_datasets.map(
        preprocess_func,
        batched=True,
        num_proc=remain_args.dataset_num_proc,
        remove_columns=column_names, # 移除原始的 text 和 label，只保留 tensors
        desc="Tokenizing dataset",
        load_from_cache_file=False
    )
    # print("Sample from processed dataset:", processed_datasets["train"][0])
    if "train" in processed_datasets:
        print(f"[DEBUG] Train dataset loaded. Size: {len(processed_datasets['train'])}")
    else:
        print("[DEBUG] CRITICAL: 'train' key not found in processed_datasets!")
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    def log_label_stats(dataset, name):
        labels = np.asarray(dataset["labels"], dtype=np.float32)
        pos = int(np.sum(labels > 0))
        rank0_print(
            rank,
            f"[DEBUG] {name} labels n/pos/mean/std/min/max: "
            f"({len(labels)}, {pos}, {float(labels.mean())}, {float(labels.std())}, "
            f"{float(labels.min())}, {float(labels.max())})",
        )

    log_label_stats(train_dataset, "train")
    log_label_stats(eval_dataset, "eval")
    
    # 6. 初始化 Trainer
    rank0_print(rank, "Starting training...")
    callbacks = []
    if remain_args.enable_swanlab:
        if SwanLabCallback is None:
            raise ImportError("swanlab is not installed; install it or omit --enable_swanlab.")
        callbacks.append(
            SwanLabCallback(
                project=remain_args.swanlab_project,
                experiment_name=config.run_name,
                description="Iterative training for GUI Agent RM",
                config={
                    "model": model_config.model_name_or_path,
                    "lr": config.learning_rate,
                    "train_dataset_size": len(train_dataset),
                    "batch_size": config.per_device_train_batch_size,
                }
            )
        )

    config.remove_unused_columns = False
    
    trainer = RegressionRMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RegressionDataCollator(
            tokenizer=tokenizer,
            max_length=config.max_length
        ),
        compute_metrics=ComputeMetrics(),
        callbacks=callbacks,
        # label_names=["labels"],
    )

    print("[DEBUG] Trainer initialized. Starting training...")

    def last_token_value(values, attention_mask):
   
        if values.dim() == 3:
            values = values.squeeze(-1)  # (B, T)

        # 每条样本最后一个有效 token 的 index
        # attention_mask.sum(-1) gives lengths; last index = len-1
        last_idx = attention_mask.long().sum(dim=1) - 1  # (B,)
        last_idx = torch.clamp(last_idx, min=0)

        bsz = values.size(0)
        return values[torch.arange(bsz, device=values.device), last_idx]  # (B,)

    def quick_eval(model, dataset, tokenizer, max_length, n=40):
        model.eval()
        labels = np.asarray(dataset["labels"], dtype=np.float32)
        pos_idx = np.flatnonzero(labels > 0)
        neg_idx = np.flatnonzero(labels <= 0)
        rng = np.random.default_rng(42)
        sample_size = min(n, len(dataset))
        pos_quota = min(len(pos_idx), max(1, sample_size // 2)) if len(pos_idx) else 0
        neg_quota = min(len(neg_idx), sample_size - pos_quota)
        chosen = []
        if pos_quota:
            chosen.extend(rng.choice(pos_idx, size=pos_quota, replace=False).tolist())
        if neg_quota:
            chosen.extend(rng.choice(neg_idx, size=neg_quota, replace=False).tolist())
        remaining = sample_size - len(chosen)
        if remaining > 0:
            all_idx = np.arange(len(dataset))
            chosen_set = set(chosen)
            rest = np.asarray([i for i in all_idx if i not in chosen_set])
            if len(rest):
                chosen.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())
        rng.shuffle(chosen)
        batch = [dataset[int(i)] for i in chosen]

        # 用你自己的 collator（它会 pad 并返回 input_ids/attention_mask/labels）
        collator = RegressionDataCollator(tokenizer=tokenizer, max_length=max_length)
        collated = collator(batch)

        # 放到 GPU（注意：多卡下只在 rank0 做 quick_eval，且用 rank0 的当前 device）
        device = next(model.parameters()).device
        for k, v in collated.items():
            if torch.is_tensor(v):
                collated[k] = v.to(device)

        with torch.no_grad():
            out = model(
                input_ids=collated["input_ids"],
                attention_mask=collated["attention_mask"],
            )

            # 这里假设 model forward 直接返回 token-level values
            # 如果你 forward 返回的是 dict/tuple，需要按你的实际结构取出来
            values = out  # <-- 如有必要改成 out["values"] 或 out[0]

            pred = last_token_value(values, collated["attention_mask"])
            pred = pred.float().detach().cpu().numpy()

        y = collated["labels"].float().detach().cpu().numpy()

        def stat(x):
            return float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))

        mse = float(np.mean((pred - y) ** 2))
        print("[DEBUG] labels mean/std/min/max:", stat(y))
        print("[DEBUG] pred   mean/std/min/max:", stat(pred))
        print("[DEBUG] init mse:", mse)

    if rank == 0 and remain_args.quick_eval_samples > 0:
        quick_eval(
            model,
            train_dataset,
            tokenizer=tokenizer,
            max_length=config.max_length,
            n=remain_args.quick_eval_samples,
        )

    trainer.train()
    
    # trainer = RegressionRMTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=config, # 这里的 config 已经是 TrainingArguments 类型了
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=RegressionDataCollator(
    #         tokenizer=tokenizer,
    #         max_length=config.max_length
    #     ),
    #     compute_metrics=ComputeMetrics()
    # )

    # # 7. 开始训练
    # trainer.train()
    print("[DEBUG] Training finished. Saving model...")
    # print(f"[DEBUG] Save path: {config.output_dir}")
    
    # 8. 保存与评估
    rank0_print(rank, "Saving model...")
    trainer.save_model(config.output_dir)
    print(f"Saving config and tokenizer to {config.output_dir}...")

    # 2. 显式保存 Config
    # 因为 Trainer 可能只保存了 wrapped model 的权重，没保存 config
    if hasattr(model, "config"):
        model.config.save_pretrained(config.output_dir)
    elif hasattr(model.pretrained_model, "config"):
        model.pretrained_model.config.save_pretrained(config.output_dir)


    print("Config and Tokenizer saved successfully.")
    # trainer.save_state()
    print("[DEBUG] Model saved successfully.")
    rank0_print(rank, "Evaluating...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    rank0_print(rank, metrics)
    
    if remain_args.metrics_path:
        os.makedirs(os.path.dirname(remain_args.metrics_path), exist_ok=True)
        with open(remain_args.metrics_path, 'w') as f:
            metrics['model_name_or_path'] = model_config.model_name_or_path
            json.dump(metrics, f, indent=2)
