# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import Trainer
from typing import Dict, List, Optional, Sequence, Any, Union, Tuple
import numpy as np
from numpy.typing import NDArray


# @dataclass
# class ComputeMetrics:
#     r"""
#     Computes regression metrics (MSE, MAE).
#     """
#     def __call__(self, eval_preds) -> Dict[str, float]:
#         predictions, labels = eval_preds
#         # Flatten dimensions
#         predictions = predictions.squeeze()
#         labels = labels.squeeze()
        
#         mse = ((predictions - labels) ** 2).mean().item()
#         mae = np.abs(predictions - labels).mean().item()
        
#         return {
#             "eval_mse": mse,
#             "eval_mae": mae
#         }

@dataclass
class ComputeMetrics:
    def __call__(self, eval_pred):
        preds = np.asarray(eval_pred.predictions).squeeze()
        labels = np.asarray(eval_pred.label_ids).squeeze()
        mse = float(((preds - labels) ** 2).mean())
        mae = float(np.abs(preds - labels).mean())
        return {"mse": mse, "mae": mae}  # Trainer 会自动加 eval_
    
@dataclass
class RegressionDataCollator:
    r"""
    Data collator for regression tasks (Points-wise).
    Handles padding input_ids and stacking scalar labels.
    """
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # if len(features) > 0 and "labels" not in features[0]:
        #     print("[DEBUG] feature keys:", features[0].keys())
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Use tokenizer to pad inputs
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest", # or 'max_length' if preferred
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Convert labels to tensor (Float/BFloat16 depending on training config, usually Float32 is safe for targets)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        
        return batch

class ValueHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        summary_dropout_prob = getattr(config, "summary_dropout_prob", 0.1)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        
        hidden_size = getattr(config, "hidden_size", 4096)
        self.summary = nn.Linear(hidden_size, 1)
        
        # Init weights
        nn.init.normal_(self.summary.weight, mean=5e-7, std=1e-6)
        nn.init.constant_(self.summary.bias, 1e-6)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # Ensure numerical stability / match dtype
        try:
            if output.dtype != self.summary.weight.dtype:
                output = output.to(self.summary.weight.dtype)
        except:
            pass
        return self.summary(output)

class RewardModelWithValueHead(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model
        if hasattr(self.pretrained_model, "lm_head"):
            # 这一步会物理删除 lm_head，从而解决共享冲突，
            # 同时也节省显存！
            del self.pretrained_model.lm_head
        self.config = pretrained_model.config
        self.v_head = ValueHead(self.config, **kwargs)
        
        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable
        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        # Clean arguments intended for Trainer but not Model
        if "labels" in kwargs: kwargs.pop("labels")
        if "return_loss" in kwargs: kwargs.pop("return_loss")

        kwargs["return_dict"] = True
        
        if hasattr(self.pretrained_model, "model"):
            backbone = self.pretrained_model.model
        else:
            backbone = self.pretrained_model

        device = backbone.embed_tokens.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs, 
        )

        last_hidden_state = outputs.last_hidden_state
        value = self.v_head(last_hidden_state).squeeze(-1)

        if return_past_key_values:
            return (value, outputs.past_key_values)
        else:
            return value

# def preprocess_regression_dataset(
#     examples,
#     tokenizer,
#     max_length=2048,
# ):
#     """
#     Standard preprocessing for text -> label regression.
#     """
#     model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

#     for i in range(len(examples["text"])):
#         text = examples["text"][i]
#         label = float(examples["label"][i]) # Ensure float
        
#         tokenized = tokenizer(
#             text, 
#             add_special_tokens=False, # Depends on your prompt format
#             padding=False, 
#             truncation=True,
#             max_length=max_length
#         )
        
#         model_inputs["input_ids"].append(tokenized["input_ids"])
#         model_inputs["attention_mask"].append(tokenized["attention_mask"])
#         model_inputs["labels"].append(label)

#     return model_inputs

def preprocess_regression_dataset(examples, tokenizer, max_length=2048):
    """
    Vectorized preprocessing for text -> label regression.
    """
    # 1. 一次性 tokenize 整个 batch（高效且结构自然正确）
    model_inputs = tokenizer(
        examples["text"],  # 这是一个 list of strings
        add_special_tokens=True,  # 建议改为 True，Qwen 需要 EOS token
        padding=False,  # padding 留给 collator
        truncation=True,
        max_length=max_length
    )
    
    # 2. 添加 labels（确保是浮点数列表）
    model_inputs["labels"] = [float(label) for label in examples["label"]]
    
    return model_inputs
    
# class RegressionRMTrainer(Trainer):
#     r"""
#     Trainer with MSE Loss for Regression.
#     """
#     def compute_loss(
#         self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        
#         # Pop labels because your custom model doesn't accept them in forward()
#         labels = inputs.pop("labels")
        
#         # Forward pass
#         # output is [batch_size] scalar values
#         outputs = model(**inputs)
        
#         # Compute MSE Loss
#         # Ensure outputs and labels are same dtype and shape
#         predictions = outputs.squeeze()
#         targets = labels.squeeze().to(predictions.device).to(predictions.dtype)
        
#         loss = F.mse_loss(predictions, targets)

#         if return_outputs:
#             return loss, outputs
#         return loss

class RegressionRMTrainer(Trainer):
    r"""
    Trainer with MSE Loss for Regression, handling sequence outputs correctly.
    """
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # 1) 不要 pop！否则 eval 阶段拿不到 label_ids
    #     labels = inputs["labels"]

    #     # 2) 给模型的输入用一个新的 dict（剔除 labels / factor 等非 forward 参数）
    #     model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "factor"]}

    #     attention_mask = model_inputs.get("attention_mask")
    #     outputs = model(**model_inputs)  # outputs: [B, L] (你的 RM 输出每 token value)

    #     # 3) 取最后一个有效 token 的分数 -> [B]
    #     if attention_mask is not None:
    #         last_idx = attention_mask.sum(dim=-1, keepdim=True) - 1  # [B,1]
    #         predictions = outputs.gather(dim=-1, index=last_idx).squeeze(-1)  # [B]
    #     else:
    #         predictions = outputs[:, -1]  # [B]

    #     targets = labels.to(predictions.device).to(predictions.dtype).squeeze()
    #     # print(predictions, targets)
    #     loss = F.mse_loss(predictions, targets)

    #     if return_outputs:
    #         # 推荐用 dict 显式提供 logits，Trainer 更好解析
    #         return loss, {"logits": predictions}
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"].float()  # [B], in [0,1]

        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels"]}
        attention_mask = model_inputs.get("attention_mask")

        outputs = model(**model_inputs)  # [B, L] token-level logits/values

        if attention_mask is not None:
            last_idx = (attention_mask.long().sum(dim=-1, keepdim=True) - 1).clamp_min(0)  # [B,1]
            logits = outputs.gather(dim=-1, index=last_idx).squeeze(-1)  # [B]
        else:
            logits = outputs[:, -1]  # [B]

        targets = labels.to(device=logits.device, dtype=logits.dtype).squeeze()  # [B]

        # (2) 类不平衡权重：soft 版本（比 labels>0 更平滑）
        # y 越大权重越大；alpha 先从 3~5 试
        alpha = 4.0
        w_pos = 1.0 + alpha * targets               # y=0 ->1, y=0.95 -> 4.8

        weights = w_pos

        # ---- BCE with logits (soft labels OK) ----
        per_example = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B]

        # 加权平均（关键）
        loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)

        if return_outputs:
            return loss, {"logits": logits}
        return loss