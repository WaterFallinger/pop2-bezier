# 模型权重与版本

当前公开推理权重是 **Primitive Operation Painter 2 — 1B**，内部模型项目版本 v3.3，Epoch 60，global step 10020。应用版本 0.9.29 与模型的训练 epoch 是两个独立版本概念。

- 仓库：https://huggingface.co/doubixz/primitive-operation-painter-2-1b
- 固定提交：`ca575269a461c90ecf630dcf70c4beaaf319b3a1`
- `model.pt`：2,008,472,552 字节，BF16，原始 `model_state_dict` 格式。
- `caption_spm_1500.model`：训练使用的 1500 词表 SentencePiece 模型。
- 权重与原始训练数据均不放入 Git 源码历史。

从发布目录根运行：

```sh
python -m pip install "huggingface_hub>=0.34,<2"
python tools/download_model.py
```

默认下载到 `weights/pop2-1b-epoch60/`。不需要 Hugging Face 账号或令牌；网络必须能访问该仓库及其文件下载服务。重复运行会利用下载缓存。下载后会核对权重与分词器的文件大小及 SHA-256。只检查已经下载的文件可运行：

```sh
python tools/download_model.py --verify-only
```

App 也可在设置中独立下载相同固定版本，或者选择已下载的本地文件。

该包是推理权重，不包含完整 Adam 状态和各 rank 的 FP32 master / optimizer 分片。它能用于模型推理；若用于继续训练，需要按训练入口明确选择“重新初始化优化器”的方式，不能把它描述成精确恢复原训练进度。精确中断恢复应使用本发布训练程序新保存的完整检查点。

早期 POP 1 权重位于 `doubixz/primitive-operation-painter-weight`。其 0.3B / 144 步模型的词表、结构和加载器与本代不兼容。
