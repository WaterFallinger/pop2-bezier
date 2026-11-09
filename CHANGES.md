# ex/ 贝塞尔图元改造工作区

原项目 `../primitive-operation-painter/` 保持只读，本目录是独立改造副本。

## 目标

在规则图形（矩形/椭圆）的参数化上增加 **4 条边的贝塞尔控制点**，让色块可以弯曲，
细长外框 + 弯曲 = 可表达笔锋/发丝/衣纹等"线"结构。随后用新数据重训模型。

## 协议变更（v3）

每个图元：**9 → 13 个离散 token**

| 字段 | bins | 偏移 | 说明 |
|---|---|---|---|
| x | 1024 | 0 | 半像素 |
| y | 1024 | 1024 | 半像素 |
| angle | 270 | 2048 | 1/3 度 |
| width | 2048 | 2318 | 1/4 像素 |
| height | 2048 | 4366 | 1/4 像素 |
| shape | 256 | 6414 | 0=背景 1=矩形 2=椭圆 |
| red/green/blue | 128×3 | 6670/6798/6926 | ×2 → 0-254 |
| **edge0..edge3** | **256×4** | **7054/7310/7566/7822** | 归一化 -1..1，0 = 直边 |
| special | 4 | 8078 | |
| empty | 12 | 8082 | y..blue + edge0..3 |
| true_end / interrupt_end / BOS | 1×3 | 8094/8095/8096 | |

- 属性词表：7054 → **8078**
- 输入词表：7069 → **8097**
- 每图元 token 数 `TOKENS_PER_STEP`：9 → **13**
- `CONTEXT_TOKENS`：8064 → 11661（位置数不变，仍是 897）

## 已改文件

| 文件 | 改动 |
|---|---|
| `model/token_layout.py` | `edge_bins/edge_count`、`edge_offset(s)`、`special_offset` 顺移、`field_slice` 支持 edge、`TOKEN_LAYOUT_VERSION_BEZIER`、断言 8078/8097 |
| `model/block_state.py` | `ATTRIBUTE_NAMES` 加 edge0..3；offsets/bins 改为经 `field_slice` 统一推导 |
| `model/config.py` | `TOKENS_PER_STEP = ATTRIBUTES_PER_PRIMITIVE`（13） |
| `model/dataset_gpt_full_sequence.py` | CSV 可选 `e0..e3` 列 → 量化进 edge token；**无该列时自动按 0 偏移（兼容旧数据）** |
| `model/reference_visualize.py` | `encode_primitive_tokens(..., edge_values=)`；`decode_tokens_to_render_data` 按 13 字段解析并输出 edge；`_composite_shape` 解包前 9 项 + 读 edge（几何渲染待实现） |
| `model/infer.py` | vocab 7069→8097、prefix `[T,9]`→`[T,13]`、protocol 校验、`reshape(-1,9)`→`(-1,13)` |
| `model/checkpoint_compat.py` | **新增 `load_bezier_initialization`**：7069 词表权重 → 8097 词表的精确映射（属性直连、edge 段零初始化、special/empty/控制 token 重定位） |
| `model/train_gpt_pretrain.py` | 加载分支支持 `POP_BEZIER_INIT=1` 走 bezier 映射（默认仍走 strict 校验） |
| `model/fused_multilane.py` | 无 Triton 时自动回退到纯 PyTorch `reference_attention`（本机 Windows 无 triton 可用） |
| `data_tools/.../shader.wgsl` | `GpuShape` 加 `e0..e3`；`get_coverage` 加贝塞尔闭式距离（δ=0 走原矩形路径）；初始化/变异加边控制点；包围盒外扩 |
| `data_tools/.../main.rs` | `EDGE_MUTATION_WEIGHT=2`、`EDGE_MUTATION_RANGE=0.20`、ShaderConstants 扩为 6 权重组 |
| `data_tools/.../types.rs` | `GpuShape` 加 4 个 f32（与 WGSL 字节对齐） |
| `data_tools/.../v3.rs` | CSV 加 `e0..e3` 列；`FORMAT` → `..._v2_bezier` |
| `data_tools/.../tools/render_csv.py` | 回放支持贝塞尔 coverage（与 WGSL 同公式） |

## 生成器实验（2026-11-09，RTX 4060 Laptop）

编译链路：MSYS2 GCC 16.2 + binutils 2.47 + Rust `stable-x86_64-pc-windows-gnu` + 173 个 crate 全部 vendored（cargo 的 schannel 在沙箱下不可用）。

单图序列化（512×512 合成图，seed 20260916）：

| 指标 | 贝塞尔版 | 基线（仓库旧数据） |
|---|---|---|
| 形状数 | **177** | 233 |
| 有弯曲的形状 | **144（81.4%）** | 0 |
| 平均弯曲强度 \|e\| | 0.40 | — |
| 回放 RMSE | **9.191** | 9.382 |
| 回放 MAE | **5.222** | 5.481 |
| 单图耗时 | ~10.7 s | — |

**结论**：搜索器大规模采用弯曲（81%），并用 **少 24% 的笔** 达到 **更低的重建误差**——贝塞尔提供了真实的逐笔信息增益，而非装饰。

## 已验证

1. `token_layout` 断言通过：vocab 8097 / attr 8078 / edge offsets (7054, 7310, 7566, 7822) / empty 8082..8093 / 控制 token 8094..8096；旧 64/192/256 布局不受影响。
2. `block_state`：13 属性、offsets/bins 正确、empty 12 个、partial fill 13 个。
3. `encode → decode` 往返：edge (0.3,-0.5,0,0.8) 还原为 (0.302,-0.498,0.004,0.804)，量化误差 < 1/255。
4. marker 检测、BOS/END 检测、marker 不参与渲染，均正常。
5. 小模型端到端前向：输入 `[1,8,13]` → logits `[1,8,8078]`。
6. **词表迁移单测**（`_pop2/tools/test_bezier_init.py`）：10 项检查全通过 —— 属性直连、edge 段为零、special/empty 重定位、控制 token 映射、text_wte 与主干层不受影响。
7. **真实训练链路**（官方 `--tiny`，合成数据，2 epoch）：
   - 数据形状 `(1, 11661)` = 897×13，旧 CSV（无 `e0..e3` 列）被向后兼容成 0 偏移
   - `Loss(total/attr/term)` 9.7156 → 9.7038；`LR_GPT` 1e-10 → 2e-4（warmup 生效）
   - 检查点 `epoch_1.pt` / `epoch_2.pt` 正常保存
8. **训练→推理闭环**：用 `epoch_2.pt`（0.70M 参数）跑 `infer.py --max-records 40`，成功输出 `001.npy`（40×13）与 `001.png`。

## 待做

| # | 任务 | 位置 | 阻塞 |
|---|---|---|---|
| 1 | **贝塞尔四边形渲染**（4 条二次贝塞尔 → 细分 → 多边形 SDF） | `reference_visualize._composite_shape` | 无，可立即做 |
| 2 | 训练脚本：vocab 传入、warm start（旧行保留 + edge 零初始化）、`RAND_K=3`、8-bit Adam | `model/train.py`、`train_gpt_pretrain.py` | 无 |
| 3 | 生成器：edge 变异算子、Δ-SSE 评分含弯曲、CSV 加 `e0..e3`、`MAX_ASPECT_RATIO 8→32`、尺寸下限 2px | `data_tools/image_to_sequence/src/{shader.wgsl,main.rs}` | 需 MSVC 链接器才能编译验证 |
| 4 | App（Rust candle）同步 13 字段与贝塞尔渲染 | `app/src/{model,canvas}` | 同 3 |
| 5 | 数据生成 + 训练 | — | 依赖 1-3 |

## 运行

```powershell
$env:PYTHONPATH="D:\DS_workspaceNew\DS_Studynew\ex\model"
python -c "import token_layout, block_state, reference_visualize"   # 冒烟
```

> 现有 1B 权重（7069 词表）与新布局不兼容，需重训；旧权重仅可用作 warm start 的初始化来源。
