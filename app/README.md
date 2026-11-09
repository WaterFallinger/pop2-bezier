# Primitive Painter Windows App

原生 Windows 图元绘画与编辑应用，版本 **0.9.29**。它是 POP 2 模型的交互入口；模型结构、训练和数据生成研究请从[项目首页](../README.md)开始。

界面和状态树由 Rust / egui / WGPU 实现；模型计算使用 Rust / Candle CUDA、C++ ATen/HIP 或 CPU 兼容后端。运行时不需要 Python，不包含浏览器服务或 WebView。

## 用户使用

安装程序以项目 Release 附件单独分发，源码目录不包含安装器成品、大型运行库或模型权重。

[下载 Windows 安装程序 0.9.29](https://github.com/wonderfulearth/primitive-operation-painter/releases/download/pop2-1b/PrimitivePainter-0.9.29-Setup.exe) · [POP 2 发布说明](https://github.com/wonderfulearth/primitive-operation-painter/releases/tag/pop2-1b)

安装后进入“设置 → 模型与运行环境”，点击在线下载，或者选择已有的 `model.pt`。程序使用的公开权重与分词器：

**[doubixz/primitive-operation-painter-2-1b](https://huggingface.co/doubixz/primitive-operation-painter-2-1b)**

这是 POP 2 / v3.3 的 1B BF16 Epoch 60 权重，与第一代 144 步模型不兼容。公开下载不需要 Hugging Face 账号；网络必须能访问该站点及其下载服务。下载固定到已验证的仓库提交，支持暂停续传和 SHA-256 校验。安装包内含小型分词器，方便提示词验证；不含 2 GB 权重。

## 当前能力

- 原生矩形、椭圆和背景编辑；选择、穿透选择、框选、多选、移动、缩放、旋转、颜色、复制粘贴、图层顺序。
- 文本、图元前缀或两者共同输入，模型逐图元续画；并行生成形成独立状态树分支。
- 可跳转、分叉、删除子树的状态树，节点缩略图、岔点优先显示和全树小地图。
- 全图、4×4 粗格和 8×8 细格阶段；区域进度、人工修改、局部重绘和区域拖动反选。
- Rényi 自适应采样、CFG、自动种子、上下文重打包、前缀 KV 复用和按能力选择的 KV INT8。
- 图像参考裁剪与原生序列化、手动存档（Ctrl+S）、PNG / SVG 导出。
- NVIDIA / AMD / CPU 自适应路由、可选优化准备与兼容提示、可导出诊断日志。

## 已验证范围

Windows 11 x64 是本轮实测平台。NVIDIA 端验证过 RTX 5070 Ti Laptop；AMD Windows 端验证过 RX 7900 XTX。无受支持 GPU 的干净 Windows 沙盒完成过安装、CPU 兼容选择和 DX12/WARP 软件界面验收；启用虚拟 GPU 后，也完成过真实 1B CUDA 单分支和 8 分支短生成。

这些结果不等于所有 Windows 版本、所有显卡均可达到相同速度。NVIDIA 通用 worker 编入 sm_80 PTX，基础要求为适用的计算能力 8.0+ 设备与支持 CUDA 13.1 的驱动；sm_120a worker 只供匹配的 12.0 设备。AMD 需要匹配的 HIP/ATen 原生运行组件并通过实际设备检查。CPU 回退速度较慢且需要足够内存。界面可通过 DX12/WARP 显示，不代表模型在 GPU 上计算。

普通用户无需安装开发工具包。程序按所选后端准备固定版本运行库；网络或能力不可用时会明确显示回退原因，而非宣称硬件加速成功。

## 源码布局

- `src/`：共享界面、状态树、区域协议、模型执行、CUDA/HIP 内核、采样、日志和模型下载。
- `backends/cpu/`：不启用 CUDA 的原生前端清单与测试资产。
- `native/aten_hip/cpp/`：Windows AMD C++ 模型和 FFI 桥接源码。
- `native/aten_hip/gui/`：AMD 原生前端 Rust 清单，与其他前端共用 `src/`。
- `vendor/candle-core/`：上游 Candle 0.11.0 源码副本；本地改动只用于 cudarc 动态链接策略，保留上游许可。
- `assets/`：必需的 SentencePiece 分词器与提示词模板；不含训练图像。
- `installer/`：固定版本运行库清单、NSIS 安装脚本及参数化打包入口。
- `scripts/build.ps1`：统一的开发构建入口。

## 从源码构建

需要 Rust 的 `x86_64-pc-windows-msvc` 工具链、Visual Studio C++ Build Tools 和 Windows SDK。本轮开发使用 Rust 1.94；依赖由锁文件固定。首次构建需要联网获取 crates，或由维护者预先准备合法依赖缓存。后续可加 `-Offline`；源码不携带开发机 Cargo 缓存。

在本目录执行：

```powershell
# CPU 应用和启动器/运行库管理工具
./scripts/build.ps1 -CpuOnly
./scripts/build.ps1 -ToolsOnly

# CPU 逻辑、序列协议、画布和 UI 回归
./scripts/build.ps1 -CpuOnly -Test

# NVIDIA 两种 worker；开发构建需要 CUDA Toolkit 13.1
./scripts/build.ps1 -CudaProfile portable -CudaToolkitRoot 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1'
./scripts/build.ps1 -CudaProfile optimized -CudaToolkitRoot 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1'

# AMD Rust 前端；C++ 桥接见下节
./scripts/build.ps1 -AmdOnly
```

可通过 `-TargetDirectory`、`-Jobs`、`-DebugBuild`、`-CheckOnly` 和 `-Bin` 控制构建。不设置或修改全局 PATH；开发环境变量只影响当前构建进程。CUDA 构建会核对工具链版本和实际 PTX target，不能把同一个 worker 重命名为另一种 profile。

## AMD C++ 桥接

使用与运行清单匹配的 **ROCm/HIP 7.1 系列 Windows 开发环境**与 **ATen/LibTorch 2.9.0 ROCm 构建**。标准的 CPU 或 NVIDIA PyTorch 包不能替代。构建需要头文件和 `.lib`，自动运行库下载器只提取用户运行所需文件，不能充当完整开发 SDK。

```powershell
./native/aten_hip/build-bridge.ps1 `
  -TorchRoot 'D:/dependencies/torch' `
  -RocmCoreRoot 'D:/dependencies/_rocm_sdk_core' `
  -HipSdkRoot 'C:/Program Files/AMD/ROCm/7.1'
```

三个位置是调用者自己的依赖目录示例；脚本没有测试电脑路径。默认使用 HIP SDK 的 `clang-cl.exe`，可用 `-CompilerPath` 更改。输出在 `target/aten-hip-bridge/`，构建 ID 从桥接源码生成，运行时性能缓存与该 ID 关联。脚本不要求 SSH 或另一台开发机。

## 制作完整安装包

见 [installer/README.md](installer/README.md)。需要先构建四个前端、启动器、运行库 helper、AMD 桥接，并提供 NSIS 3.x 和已获许可的 Microsoft VC x64 CRT 再分发文件目录。源码不附带 NSIS 可执行文件、VC DLL、CUDA/HIP DLL 或历史 staging 产物。

## 诊断与许可

设置页可打开日志和导出诊断。显示适配器清单只是路由线索；模型实际设备应以 `model_compute_context` 为准，界面设备记录为 `ui_renderer.selected`。默认不写入提示词正文、原始图元数组、权重或凭据。程序不自动上传日志。

第一方应用代码遵循 MIT，见 `LICENSE`。第三方依赖分别遵循各自许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。训练数据与大型模型权重不随源码发布。
