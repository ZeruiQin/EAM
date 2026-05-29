from transformers import AutoTokenizer

model_path = "/root/autodl-tmp/rStar-rStar-math/qwen/Qwen2.5-3B-Instruct"  # 替换你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 你的 Prompt 结尾是 "Answer: This action is" (末尾没有空格)
# 所以下一个词必须包含前导空格
candidate_pairs = [
    (" helpful", " unhelpful"),
    (" beneficial", " detrimental"),
    (" good", " bad"),
    (" correct", " incorrect"),
    (" right", " wrong"),
    (" Yes", " No"),
    (" yes", " no")
]

print(f"Model: {model_path}\n")
print(f"{'Positive':<15} | {'IDs':<15} | {'Len':<5} || {'Negative':<15} | {'IDs':<15} | {'Len':<5}")
print("-" * 80)

valid_pair_found = False

for pos, neg in candidate_pairs:
    # 关键：add_special_tokens=False，只看单纯的词
    pos_ids = tokenizer.encode(pos, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg, add_special_tokens=False)
    
    pos_len = len(pos_ids)
    neg_len = len(neg_ids)
    
    print(f"'{pos}'".ljust(15) + f" | {str(pos_ids):<15} | {pos_len:<5} || " + 
          f"'{neg}'".ljust(15) + f" | {str(neg_ids):<15} | {neg_len:<5}")

    if pos_len == 1 and neg_len == 1:
        valid_pair_found = True
        best_pos = pos
        best_neg = neg

print("-" * 80)
if valid_pair_found:
    print(f"✅ 推荐使用单 Token 词对: '{best_pos}' / '{best_neg}'")
    print(f"它们可以直接用於优化版的 rm.py，显存占用最小。")
else:
    print("❌ 没有找到单 Token 词对。")
    print("Qwen2 等大词表模型通常 ' helpful' 是单 token，' unhelpful' 可能是两个 (un + helpful)。")