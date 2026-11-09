# 自备文本标注与多版本提示词

本目录说明如何给图像序列配套文本，附带的 JSON 只描述仓库内程序绘制的合成色块示例，不来自原训练集。这里不包含 LLM 服务、账号配置、原始全量提示词或外部模型权重。

图像序列化和文本标注是两个独立步骤：`image_to_sequence` 负责图形和区域控制记录，标注者为每张图提供文字，`model/prepare_data.py` 将二者按图像名称关联并生成训练用文字索引。文本可由自己编写或自行选择的标注工具产生；无须使用某个指定商业 LLM。

## 输入合同

`--captions` 接受 UTF-8 JSON 对象，键为 CSV 中**不带扩展名的 `image_name`**。值可以是一个提示词字符串，或非空的字符串数组。

```json
{
  "smoke_000001": [
    "彩色的背景。",
    "背景有不同颜色的图形。"
  ]
}
```

文件 `captions.synthetic.example.json` 可以直接与 `examples/synthetic/sequences` 配套。它有一张图、两个文字版本；用于数据链路验收，不代表有意义的图像模型训练语料。

必须满足：

1. JSON 键集合与所有选用 `vN/.completed_images.txt` 中的图像名称完全相同，不能缺图、混入额外图片或携带文件扩展名。各几何版本也必须具有相同完成图像集合。
2. 每张图的文字版本数量必须是相同的正整数。所有图片都只有一个版本时，值可直接用字符串；有多版本时，建议每张图都用等长数组。数组顺序就是一开始的文字版本编号 1、2、3……。
3. 普通文字标注不能为空。每段文本经当前分词器编码后，加 BOS/EOS 的总长度不得超过 **224 token**；超长输入会被拒绝，原文不会静默截断或跳过。
4. 当前公开权重需要随发布提供的原版 **1500 词表 SentencePiece**；新训练一个分词器会改变文字嵌入合同，不能当作公开权重的直接替换。建议先检查所用文字能被该词表合理表达。

多版本应描述同一幅图，可以改变取景描述、属性详略、句式或风格表述，但不应凭空新增图中不存在的内容。自动标注的输出也需要对照图像抽查。目录路径、任务编号或 API 提示模板不应作为给绘画模型的描述文字。

## 准备文字缓存

从发布目录根运行：

```sh
python -m pip install -r model/requirements.txt
python model/prepare_data.py --data-dir examples/synthetic/sequences --captions data_tools/captions/captions.synthetic.example.json --output work/synthetic-captions
```

准备器会验证全部文字，生成 `selected_images.tsv`、按图像编号组织的 `prompts/`、紧凑的 `cache/` 和 `prepared.json`。输出目录必须为空或尚不存在；旧缓存不会被覆盖。编号由准备器生成，公开使用者不需要复制开发者的选择清单或队列数据库。

在支持的训练环境中，可以用同一份缓存做小模型链路验证：

```sh
python model/train.py --data-dir examples/synthetic/sequences --caption-root work/synthetic-captions --checkpoint-dir work/tiny-checkpoints --versions 1 --epochs 2 --micro-batch 1 --accumulation 1 --tiny --from-scratch
```

训练入口默认从 `prepared.json` 推断文字版本数量；`--caption-variants` 仅在明确需要覆盖时指定，并须与缓存一致。CPU 数据准备和协议校验不需要加载 1B 权重；上面的训练命令仍需要 `model/README.md` 所述的适当训练环境。

## 几何版本和文字版本如何使用

两者不需要数量相等。例如可以准备 50 个几何版本，以及每张图 100 个文字版本。

- 几何版本按 epoch 循环：`((epoch - 1) % versions) + 1`。`train.py --versions N` 决定这一轮可用的几何版本数量，不是文字数量。
- 文字由 `caption_data.py` 根据种子、epoch 和图像所在行计算确定性的伪随机选择。同一设定下，同一张图在同一个 epoch 的文字选择可以复现，跨 epoch 可以换到别的文字版本。
- 默认还按 **30%** 的概率使用训练文本 dropout：实际条件成为 `[BOS, EOS] = [2, 3]`。这为后续有文字/无文字的 CFG 比较提供空文字分支。
- 多个文字版本不会同时拼到一个模型前缀中；总上下文仍然是 1024。有限 epoch 数不保证每张图的每个文字版本都恰好被用遍。

若只希望做无文字训练，需要明确选择：

```sh
python model/prepare_data.py --data-dir examples/synthetic/sequences --unconditional --output work/synthetic-no-text
```

该方式直接写入正确的空文字 token 序列，不应把“无提示词”“UNCONDITIONAL”等文字当作描述交给模型。

## 与原研究数据的边界

原研究中曾使用过本地 Qwen 属性分析/改写流程，后来还使用了 Luna 辅助生成和重排的多版本提示词。那些历史工作目录涉及不同的流程、外部模型和任务队列，不能把任意旧脚本标成“最新 100 版本标注器”。本发布给出独立的数据合同与读取代码，让研究者可以用自有数据和标注来源复现实验；不会声称已经公开原始完整语料或特定外部服务。
