# POP 2 图像 → 图元序列数据生成工具

这是 POP 2 / v3.3 训练所用的三阶段 GPU 图像序列化程序，独立于桌面 App。它用矩形、椭圆逐步逼近用户提供的图像，并将真实绘制顺序和区域控制信息写成训练用 CSV；不需要模型权重、PyTorch 或 Python 来执行序列化。

本目录保留正式数据生成器的 WGSL 搜索、解析颜色求解、变异爬山、调度和断点写盘算法。发布副本仅新增命令行入口，移除开发机输入/输出默认路径，将执行规模改为小样本默认，并附上独立 CSV 回放工具。App 的“参考图导入”另有 RMSE 达标后发布暂停 marker 的后处理，不能将它的结果替代本工具的训练数据。

## 构建与快速运行

需要 Rust 1.89 或更新版本，以及系统上可工作的硬件图形计算驱动。本程序不使用 CUDA/HIP API：Windows 使用 DX12、Linux 使用 Vulkan、macOS 代码路径使用 Metal。macOS 未在本次整理中进行硬件验收。CPU 软件适配器会被明确排除。

```sh
cargo build --locked --release
cargo run --locked --release -- --help
cargo run --locked --release -- --input ./my_images --output ./sequences --images 16 --versions 1 --gpus 1
```

Windows 编译需要 Rust 的 MSVC 工具链和 Visual Studio C++ Build Tools；Linux 需要通常的 Rust 链接构建工具与 Vulkan 驱动。普通用户仅运行预编译 App 时，不需要这些开发工具。

必须显式提供 `--input` 和 `--output`，或者设置相应的环境变量；程序不会扫描开发者硬盘中的默认数据集。不提供参数、路径不合法或使用 `--help` 时，在创建设备和绘制之前返回。输入和输出不能是同一个目录。请使用专用输出目录。

输入目录递归查找受支持图像，文件名主干用于 `image_name`，因此各图必须有唯一主干。输入会以 Lanczos3 缩放为 512×512；非正方形输入会被拉伸，应先自行裁剪为正方形。该发布不附带任何训练图像或提示词数据。

## 执行参数

命令行参数优先于同名环境变量。默认值只控制工作量，不改变模型的数据协议。

| 参数 | 默认值 | 对应环境变量 |
|---|---:|---|
| `--input DIR` | 必填 | `SHAPE_RENDERER_INPUT_DIR` |
| `--output DIR` | 必填 | `SHAPE_RENDERER_OUTPUT_DIR` |
| `--images N` | 16 | `SHAPE_RENDERER_IMAGE_LIMIT` |
| `--versions N` | 1 | `SHAPE_RENDERER_NUM_VERSIONS` |
| `--gpus N` | 1 | `SHAPE_RENDERER_GPU_COUNT` |
| `--batch-size N` | 16 | `SHAPE_RENDERER_IMAGES_PER_GPU_BATCH` |
| `--dispatch-size N` | 16 | `SHAPE_RENDERER_IMAGES_PER_GPU_DISPATCH` |
| `--in-flight N` | 1 | `SHAPE_RENDERER_MAX_IN_FLIGHT_SUBMISSIONS` |
| `--mode MODE` | mixed | `SHAPE_RENDERER_MODE` |
| `--seed N` | 20260916 | `SHAPE_RENDERER_SEED` |

`--batch-size` 控制每卡驻留的数据量；`--dispatch-size` 控制每次提交覆盖的图像数；`--in-flight` 控制尚未等待完成的提交数量。这三个参数是独立的。首次运行应保持小批量；增加并行量前，根据真实设备限制和显存检查结果调整。降低 dispatch 数量并不会降低整批驻留像素缓冲区的容量。

请求 GPU 数超过实际成功初始化数量时会报错。启动日志会列出各硬件适配器和原生后端。当前程序在发现阶段初始化候选设备，然后选取指定数量进行绘制；设备编号以日志为准。

## 三阶段生成规则

- 画布为 512×512，第一行是原图 RGB 均值背景；背景不计入下面的普通图形配额。
- 全图阶段成功绘制 **85** 个普通图形。
- 粗网格阶段为 **4×4**，每块最多 **20** 个普通图形。
- 细网格阶段为 **8×8**，每块最多 **10** 个普通图形。
- 区域 RMSE 阈值为 **12.75**，即 `1 - RMSE/255 >= 0.95`。
- 区域限制作用于图形中心，图形边缘允许跨格。
- 每条 CSV 序列最多 **896 条记录，包含背景和全部 marker**；表头不计，模型 BOS 不在 CSV 内。
- 记录预算耗尽属于有效截断，允许最后一个区域没有结束 marker、允许细阶段尚未开始；不能将截断补成“该区域永久完成”。

`mixed` 为正式的随机流程：每张图、每个版本独立选择普通/扰动流程，概率 30%/70%。扰动流程为两个阶段预先分别选取 0–3 个粗格和 0–12 个细格；每个成功普通图形后，若该格尚未正常结束，重新采样干预概率并决定是否暂停换格。永久结束优先，重入不清零累计图形数量。`normal`、`perturbed` 用于明确指定某种流程的实验。

原生图形透明度固定为 0.5；颜色由当前画布和候选覆盖区域解析求解。默认保留 50 个初始候选、5 个幸存候选、连续 100 次变异无改善后停止、每候选 10000 次变异硬上限，以及最大长宽比 8 等原搜索配置。`src/shader.wgsl`、`src/types.rs` 和算法核心保持正式源文件。

除常用 CLI 参数外，源码还保留历史实验用 `SHAPE_RENDERER_*` 高级选项；改变区域配额、全局长度、RMSE 或记录长度时，必须确认训练配置和新输出目录与之匹配。与当前公开权重兼容的数据应保留上述 85/20/10 规则。

## 输出格式与断点

```text
sequences/
  manifest.json
  .run.lock
  v1/
    data_part_1.csv
    data_part_2.csv
    .completed_images.txt
  v2/ ...
```

存储布局标识为 `large_csv_v1`，每个 `data_part_N.csv` 约 1 GiB 封顶，每张图的记录连续写入，不跨文件拆开。`.completed_images.txt` 记录该版本已提交的图像主干。当前正式 packed 输出不是旧版的“每图一个 sequence.csv + summary.json”目录结构。

运行锁防止同时写入同一输出目录。一个批次按“CSV 写入同步 → 完成清单同步 → 提交事务”处理；中断后保留的 `.csv_write_transaction` 会在续跑时回滚到上次完整提交位置。仅识别该生成器所拥有的输出文件。不认识的非空输出目录会被拒绝。

续跑使用相同输入图像、种子、模式、版本数量和绘制协议，重新执行原命令即可；执行批量和提交大小允许调整。`manifest.json` 不匹配时应使用新目录。不要删除 `.completed_images.txt` 或手工拼接未完成的 CSV。搜索耗尽 `search_exhausted` 不作为合法训练样本提交；达到记录预算的 `record_limit` 是有效结果。

## CSV 协议

表头：

```csv
image_name,cx,cy,w,h,shape_type,theta,r,g,b,is_region_marker,stage,region_id
```

- `cx,cy,w,h` 是 512 画布像素坐标；`w,h` 为完整宽高，`theta` 为弧度。
- `r,g,b` 为 0–255 颜色；第一行背景 `shape_type=-1`。
- 普通图形 `shape_type=0` 为旋转矩形，`1` 为椭圆。
- marker 使用 `is_region_marker=1`，不绘制像素：`2` 是区域开始，`3` 是永久结束，`4` 是干预/暂停结束。
- `stage` 为 `global/coarse/fine`；全图和背景 `region_id=-1`，粗格 0–15、细格 0–63，均从左上角按行编号。
- 暂停后允许再次开始同一区域；永久结束后不能再次进入该区域。
- 数据语义标识为 `geometrize_512_three_stage_interrupt_v1`；训练读取器负责映射到当前模型词表和终止头。

提示词是另一个输入源，通过 `image_name` 与序列关联，不会写入图像 CSV。本工具既不调用外部语言模型，也不要求 OpenAI/Hugging Face 登录。

## 独立回放

回放只依赖导出的 CSV，不读取原图或模型权重。区域 marker 不绘制；背景、SDF 抗锯齿、alpha=0.5 及逐步整数颜色截断沿用正式绘制规则。

```sh
python -m pip install -r tools/requirements.txt
python tools/render_csv.py --csv-root ./sequences --version v1 --output ./preview --limit 16
python tools/render_csv.py --csv-root ./sequences --version v1 --output ./one_preview --image-name my_image --limit 1
```

生成 PNG 和 `replay_summary.json`。`--max-primitives N` 可仅回放前 N 个普通图形。工具默认最多读取前 16 个图像组，避免将完整训练集载入内存。它检查基本数值和字段，属于视觉回放工具；完整训练协议合法性以配套训练读取器为准。

## 测试与许可

```sh
cargo test --locked --release
python -m unittest discover -s tools -p 'test_*.py' -v
```

默认 Rust 测试检查 CLI、布局、调度、记录预算、扰动采样及断点写盘。依赖真实 GPU 的测试默认忽略，可以在明确空闲设备上用 `cargo test --locked --release -- --ignored --nocapture` 执行。发布整理的短测使用程序生成的合成图，不分发外部训练数据。

MIT，见 `LICENSE`。Rust 依赖遵循各自许可证。原始训练图像、检测/标注模型权重和用户数据不随本工具重新分发。`SOURCE_PROVENANCE.json` 记录算法与发布入口变更范围。
