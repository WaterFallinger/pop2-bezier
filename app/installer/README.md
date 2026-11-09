# Windows 安装包构建

`build-universal.ps1` 打包当前应用版本 **0.9.29**。安装包包含适配 NVIDIA / AMD / CPU 的原生应用，模型权重和大型运行库分开下载。

## 前置产物

从 `app/` 执行根 README 的构建步骤。默认预期路径：

| 产物 | 路径 |
|---|---|
| 启动器、两个 runtime helper | `target-tools/release/` |
| CPU worker | `target-cpu/release/primitive-painter-cpu.exe` |
| NVIDIA portable / optimized worker | `target-cuda-portable/release/primitive-painter.exe` / `target-cuda-optimized/release/primitive-painter.exe` |
| CUDA profile 元数据 | 两个 target 目录内的 `cuda-build-profile.json` |
| AMD worker | `target-aten/release/primitive-painter-aten-hip.exe` |
| AMD C++ DLL | `target/aten-hip-bridge/painter_aten_hip.dll` |

所有路径均可用脚本对应参数显式覆盖。打包前检查 PE x64、两个 CUDA worker 的构建 profile 及运行清单。没有这些产物时会报错；不会从开发电脑历史安装目录偷偷补齐。

## 外部打包依赖

- NSIS 3.x。通过 `-NsisCompiler` 指定 `makensis.exe`，或设置 `NSIS_COMPILER`，或将其放入构建进程的 PATH。
- 合法的 Microsoft Visual C++ x64 CRT 再分发文件目录。通过 `-CrtDirectory` 或 `PAINTER_VC_REDIST_DIR` 显式指定，例如已安装 Visual Studio 的 `VC/Redist/MSVC/<version>/x64/Microsoft.VC143.CRT`。不要使用任意网站下载的同名 DLL。
- 上述依赖的再分发须满足各自许可；这些工具及 DLL 不放入源码仓库。

示例（路径按本机安装位置替换）：

```powershell
./installer/build-universal.ps1 `
  -NsisCompiler 'C:/Program Files (x86)/NSIS/makensis.exe' `
  -CrtDirectory 'C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC/<version>/x64/Microsoft.VC143.CRT'
```

`-ValidateOnly` 仅验证输入；`-SyntaxCheck` 使用明确标为不可分发的占位文件检查 NSIS 语法。正式构建输出 `installer/output/PrimitivePainter-0.9.29-Setup.exe`。`staging/` 与 `output/` 均为本机构建产物，已被忽略。

## 运行组件

`runtime-manifest.json` 锁定 NVIDIA 官方再分发包和每个使用文件的 SHA-256。`native/aten_hip/native-runtime-manifest.json` 记录 AMD 官方 Windows 原生 wheel 文件和条目校验信息。用户侧 helper 不安装 Python、编译器、显卡驱动或系统 PATH。

网络不通时不能假定所有 GPU 组件已安装；应用会明确说明 CPU 回退或可选加速未启用。在线下载与硬件支持是不同的检查。

## 第三方材料

源码目录已附第一方 MIT、vendored Candle 许可，以及 `native/aten_hip/licenses/` 的 21 份 AMD/ATen/OpenBLAS 许可与告知文本。新源码的打包脚本会将这些文本安装到 `licenses/`。如果还需纳入顶层逐依赖材料，增加 `-AdditionalLicenseDirectory ../third_party`。

已经生成的原版 0.9.29 Setup **没有被重建或修改**，新增材料以发布目录侧文件提供；不能说原安装包已经包含这些文字。原生运行库 manifest 严格提取契约也保持不变，许可证由打包脚本随 App 分发。

还需核对的准确范围是：rocm-openblas 的 AMD fork 源码提交、缺少 License 元数据的两个 ROCm SDK wheel 的分发条款及全部传递依赖、AOTriton/liblzma/libomp/MIOpen 组件与现有告知的对应关系，以及所用 Microsoft VC 再分发条款。详情见 `native/aten_hip/licenses/PROVENANCE.json`。
