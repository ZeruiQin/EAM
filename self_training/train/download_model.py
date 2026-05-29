from modelscope import snapshot_download

# 下载 Qwen2.5-0.5B-Instruct 到当前目录下的 Qwen2.5-0.5B 文件夹
model_dir = snapshot_download(
    'qwen/Qwen2.5-0.5B-Instruct', 
    cache_dir='./', 
    revision='master'
)
print(f"模型已下载到: {model_dir}")