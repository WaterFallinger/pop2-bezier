param(
    [string]$Version='0.9.29',
    [string]$LauncherPath='', [string]$CudaOptimizedPath='', [string]$CudaPortablePath='',
    [string]$AtenFrontendPath='', [string]$CpuFrontendPath='', [string]$CppLibraryPath='',
    [string]$RuntimeInstallerPath='', [string]$AtenRuntimeInstallerPath='',
    [string]$NativeManifestPath='', [string]$TokenizerPath='', [string]$CrtDirectory='',
    [string]$NsisCompiler='', [string]$AdditionalLicenseDirectory='',
    [string]$CudaOptimizedMetadata='', [string]$CudaPortableMetadata='',
    [string]$PerformanceProfile='', [string]$OutputPath='',
    [switch]$ValidateOnly, [switch]$SyntaxCheck
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$project=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if($Version -notmatch '^\d+\.\d+\.\d+$'){throw 'Version must be numeric major.minor.patch'}
if($ValidateOnly -and $SyntaxCheck){throw 'ValidateOnly and SyntaxCheck are exclusive'}
function Default-Path([string]$value,[string]$relative){if($value){[IO.Path]::GetFullPath($value)}else{Join-Path $project $relative}}
$LauncherPath=Default-Path $LauncherPath 'target-tools\release\PrimitivePainter.exe'
$CudaOptimizedPath=Default-Path $CudaOptimizedPath 'target-cuda-optimized\release\primitive-painter.exe'
$CudaPortablePath=Default-Path $CudaPortablePath 'target-cuda-portable\release\primitive-painter.exe'
$AtenFrontendPath=Default-Path $AtenFrontendPath 'target-aten\release\primitive-painter-aten-hip.exe'
$CpuFrontendPath=Default-Path $CpuFrontendPath 'target-cpu\release\primitive-painter-cpu.exe'
$CppLibraryPath=Default-Path $CppLibraryPath 'target\aten-hip-bridge\painter_aten_hip.dll'
$RuntimeInstallerPath=Default-Path $RuntimeInstallerPath 'target-tools\release\runtime-installer.exe'
$AtenRuntimeInstallerPath=Default-Path $AtenRuntimeInstallerPath 'target-tools\release\aten-runtime-installer.exe'
$NativeManifestPath=Default-Path $NativeManifestPath 'native\aten_hip\native-runtime-manifest.json'
$TokenizerPath=Default-Path $TokenizerPath 'assets\caption_spm_1500.model'
# CRT files are a developer-supplied licensed redistributable directory, never a historical staging folder.
if(!$CrtDirectory){$CrtDirectory=$env:PAINTER_VC_REDIST_DIR}
if(!$CrtDirectory-and!$SyntaxCheck){throw 'Provide -CrtDirectory (the licensed Microsoft VC x64 CRT redistributable folder), or PAINTER_VC_REDIST_DIR. See installer/README.md.'}
if($CrtDirectory){$CrtDirectory=[IO.Path]::GetFullPath($CrtDirectory)}
$CudaOptimizedMetadata=Default-Path $CudaOptimizedMetadata 'target-cuda-optimized\cuda-build-profile.json'
$CudaPortableMetadata=Default-Path $CudaPortableMetadata 'target-cuda-portable\cuda-build-profile.json'
$OutputPath=Default-Path $OutputPath ('installer\output\PrimitivePainter-'+$Version+'-Setup.exe')
function Assert-SafeLiteral([string]$value){if($value.IndexOfAny([char[]]@('"','$',"`r","`n")) -ge 0){throw 'NSIS input paths cannot contain quotes, dollar signs or newlines'}}
function Assert-File([string]$path){if(!(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required input is missing: $path"}}
function Assert-PE([string]$path){
    Assert-File $path
    $stream=[IO.File]::OpenRead($path)
    try{
        $reader=[IO.BinaryReader]::new($stream)
        if($stream.Length -lt 256 -or $reader.ReadUInt16() -ne 0x5A4D){throw "Not an application PE image: $path"}
        $stream.Position=0x3c;$offset=$reader.ReadInt32()
        if($offset -lt 64 -or $offset -gt ($stream.Length-24)){throw "Invalid PE header: $path"}
        $stream.Position=$offset
        if($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0x8664){throw "Payload is not a Windows x64 PE image: $path"}
    }finally{$stream.Dispose()}
}
function Assert-CudaMetadata([string]$path,[string]$profile,[string]$target){
    Assert-File $path;$data=Get-Content -LiteralPath $path -Raw -Encoding UTF8|ConvertFrom-Json
    if($data.profile -ne $profile -or $data.embedded_ptx_target -ne $target -or !$data.embedded_ptx_verified -or $data.action -ne 'build'){
        throw "CUDA build profile is not a verified $profile build: $path"
    }
}
$sourceFiles=[ordered]@{
    'PrimitivePainter.exe'=$LauncherPath
    ('runtime/app-'+$Version+'/primitive-painter-cuda-optimized.exe')=$CudaOptimizedPath
    ('runtime/app-'+$Version+'/primitive-painter-cuda-portable.exe')=$CudaPortablePath
    ('runtime/app-'+$Version+'/primitive-painter-aten-hip.exe')=$AtenFrontendPath
    ('runtime/app-'+$Version+'/primitive-painter-cpu.exe')=$CpuFrontendPath
    ('runtime/app-'+$Version+'/painter_aten_hip.dll')=$CppLibraryPath
    'runtime/runtime-installer.exe'=$RuntimeInstallerPath
    'runtime/aten-runtime-installer.exe'=$AtenRuntimeInstallerPath
    'native-runtime-manifest.json'=$NativeManifestPath
    'assets/caption_spm_1500.model'=$TokenizerPath
}
# Include license texts with new source-built installers; no runtime binary is added here.
$sourceFiles['licenses/PrimitivePainter-MIT.txt']=Join-Path $project 'LICENSE'
$sourceFiles['licenses/Candle-Apache-2.0.txt']=Join-Path $project 'vendor/candle-core/LICENSE'
$licenseRoots=@(@{prefix='licenses/amd';path=(Join-Path $project 'native/aten_hip/licenses')})
if($AdditionalLicenseDirectory){$licenseRoots+=@{prefix='licenses/dependencies';path=[IO.Path]::GetFullPath($AdditionalLicenseDirectory)}}
foreach($licenseRoot in $licenseRoots){
    if(!(Test-Path -LiteralPath $licenseRoot.path)){throw ('License source directory missing: '+$licenseRoot.path)}
    foreach($file in (Get-ChildItem -LiteralPath $licenseRoot.path -Recurse -File)){
        if($file.Extension -notin @('.txt','.md','.json','.html')){continue}
        $relative=$file.FullName.Substring(([IO.Path]::GetFullPath($licenseRoot.path)).Length).TrimStart('\','/').Replace('\','/')
        Assert-SafeLiteral $relative
        $sourceFiles[$licenseRoot.prefix+'/'+$relative]=$file.FullName
    }
}
if(!$SyntaxCheck){
    foreach($path in $sourceFiles.Values){Assert-SafeLiteral $path;Assert-File $path}
    foreach($path in @($LauncherPath,$CudaOptimizedPath,$CudaPortablePath,$AtenFrontendPath,$CpuFrontendPath,$CppLibraryPath,$RuntimeInstallerPath,$AtenRuntimeInstallerPath)){Assert-PE $path}
    if($CudaOptimizedPath -eq $CudaPortablePath){throw 'CUDA optimized and portable inputs must be distinct build artifacts'}
    Assert-CudaMetadata $CudaOptimizedMetadata 'optimized' 'sm_120a'
    Assert-CudaMetadata $CudaPortableMetadata 'portable' 'sm_80'
    $manifest=Get-Content -LiteralPath $NativeManifestPath -Raw -Encoding UTF8|ConvertFrom-Json
    if($manifest.schema_version -ne 1 -or !$manifest.runtime_id -or @($manifest.components).Count -ne 3){throw 'The native runtime manifest is incomplete'}
    $crtFiles=@(Get-ChildItem -LiteralPath $CrtDirectory -File -Filter '*.dll'|Where-Object {$_.Name -match '^(msvcp|vcruntime|concrt)[A-Za-z0-9_.-]*\.dll$'})
    if(!($crtFiles|Where-Object Name -eq 'vcruntime140.dll') -or !($crtFiles|Where-Object Name -eq 'msvcp140.dll')){throw 'App-local Microsoft VC runtime files are incomplete'}
    foreach($file in $crtFiles){Assert-PE $file.FullName}
    if($PerformanceProfile){
        Assert-SafeLiteral $PerformanceProfile;Assert-File $PerformanceProfile
        $profile=Get-Content -LiteralPath $PerformanceProfile -Raw -Encoding UTF8|ConvertFrom-Json
        if($profile.schema_version -ne 1 -or $profile.application_version -ne $Version -or $profile.provider -ne 'aten_hip' -or $profile.build_id -notmatch '^[0-9a-fA-F]{64}$'){throw 'Performance profile does not match this application version'}
        $binary=[Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($CppLibraryPath))
        if(!$binary.Contains([string]$profile.build_id)){throw 'Performance profile build ID is absent from the staged C++ provider'}
        $sourceFiles['aten-performance-profile.json']=[IO.Path]::GetFullPath($PerformanceProfile)
    }
    if($ValidateOnly){[pscustomobject]@{status='validated';version=$Version;payload_files=$sourceFiles.Count;crt_files=$crtFiles.Count;output=$OutputPath};return}
}
$stage=Join-Path $PSScriptRoot ('staging\universal-'+$Version+'-'+[Guid]::NewGuid().ToString('N'))
$payload=Join-Path $stage 'payload'
New-Item -ItemType Directory -Path $payload -Force|Out-Null
Assert-SafeLiteral $stage
if($SyntaxCheck){
    $OutputPath=Join-Path $stage ('PrimitivePainter-'+$Version+'-SyntaxCheck-NOT-FOR-DISTRIBUTION.exe')
    foreach($relative in @($sourceFiles.Keys)){
        $destination=Join-Path $payload $relative
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force|Out-Null
        [IO.File]::WriteAllText($destination,'NSIS syntax fixture; not executable',[Text.UTF8Encoding]::new($false))
    }
    $crtFiles=@([pscustomobject]@{Name='msvcp140.dll';FullName=''},[pscustomobject]@{Name='vcruntime140.dll';FullName=''})
}else{
    foreach($entry in $sourceFiles.GetEnumerator()){
        $destination=Join-Path $payload $entry.Key
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force|Out-Null
        Copy-Item -LiteralPath $entry.Value -Destination $destination
    }
}
$crtRoot=@();$crtRuntime=@();$crtApp=@();$owned=@($sourceFiles.Keys)
foreach($file in $crtFiles){
    foreach($directory in @('','runtime',('runtime/app-'+$Version))){
        $relative=if($directory){$directory+'/'+$file.Name}else{$file.Name}
        $destination=Join-Path $payload $relative
        if($SyntaxCheck){[IO.File]::WriteAllText($destination,'syntax fixture')}else{Copy-Item -LiteralPath $file.FullName -Destination $destination}
        $owned+=$relative
    }
    $crtRoot+=('File "${PAYLOAD}\'+$file.Name+'"')
    $crtRuntime+=('File "${PAYLOAD}\runtime\'+$file.Name+'"')
    $crtApp+=('File "${PAYLOAD}\runtime\app-${VERSION}\'+$file.Name+'"')
}
function Write-Utf8([string]$path,[string]$value){[IO.File]::WriteAllText($path,$value,[Text.UTF8Encoding]::new($false))}
Write-Utf8 (Join-Path $payload 'model-config.json') '{"schema_version":1,"model_path":"models/model.pt","tokenizer_path":"assets/caption_spm_1500.model"}'
Write-Utf8 (Join-Path $payload 'aten-runtime.json') '{"runtime_dirs":["runtime/aten/torch","runtime/aten/core","runtime/aten/libraries"],"experimental_efficient_attention":true}'
Write-Utf8 (Join-Path $payload 'README.txt') @"
Primitive Painter $Version — Windows 本地原生矢量绘画

桌面“Primitive Painter”自动识别本机设备并选择对应的原生计算后端。
初次使用只准备本机需要的 NVIDIA 或 AMD 原生运行组件；没有适用GPU或准备失败时明确进入CPU兼容计算。
程序不安装Python、开发工具、显卡驱动或系统PATH，不包含大型模型权重。
在“设置”中可匿名下载官方模型或选择本地模型，并打开运行日志目录或导出诊断；默认不记录提示词或权重内容。
“Primitive Painter（兼容模式）”保留基础执行路径；CPU兼容后端速度较慢。
NVIDIA基础GPU路径面向SM8.0及以上且驱动支持CUDA13.1的适用设备；sm120a另有专用优化前端，仅匹配设备会使用。其他设备仍需通过实际运行检查。
AMD Windows路径已在RX7900XTX上实测；其他AMD设备依据实际设备、原生组件和能力自检选择路径，不承诺未经实测显卡的速度。
普通兼容模式不会绕过基础GPU支持条件；不能使用GPU时会明确告知CPU回退及原因。
排查运行环境和执行路径可先阅读同目录DIAGNOSTICS.txt。
卸载保留模型、用户设置、日志、旧版本窗口文件及已准备的大型运行组件。
画布与状态树使用手动保存（Ctrl+S）；关闭前请保存需要保留的项目。
"@
Write-Utf8 (Join-Path $payload 'THIRD_PARTY_NOTICES.txt') @'
Primitive Painter uses Rust/C++ with egui/eframe, WGPU, Candle, ATen/LibTorch, serde, image, sentencepiece-rs, zip, sha2 and their dependencies.
NVIDIA runtime components are obtained from pinned official NVIDIA CUDA redistribution packages. AMD native components are extracted from pinned official AMD ROCm Windows wheels; no Python interpreter or .pyd files are installed.
Microsoft Visual C++ runtime DLLs are app-local redistributables from the licensed build environment. Collected third-party license texts and notices are installed under licenses/. The source release records the remaining exact-binary provenance review; this notice does not relicense vendor components.
NSIS installer: zlib/libpng license. This developer preview is not code-signed.
'@
$diagnostics=@'
Primitive Painter @VERSION@ — 诊断入口

一、从哪里开始

应用“设置 → 运行日志”提供“打开日志目录”和“导出诊断”。
默认目录是当前用户LocalAppData\PrimitivePainter\logs\<会话编号>。启动器与其启动的计算前端共用会话编号，每个进程分别写文件。
events-*.jsonl是逐行JSON完整事件；summary-*.txt是便于阅读的最近事件摘要；session-*.json记录会话、应用版本和构建信息。
导出的diagnostics-*.zip汇集本会话的这些文件。导出不会打包模型权重、画布原始图元数组或提示词正文；程序也不会自动上传诊断包、隐私内容或权重。
默认目录不可写时会尝试备用临时目录；日志存储失败不会阻断模型。完全不可写时保留有限的最近内存事件。
安装目录还可能有runtime-prepare-<版本>.json和install-<版本>.log，用于查看安装与运行组件准备结果。

二、怎样解释设备和计算路径

display_inventory记录Windows显示适配器清单，仅用于选择候选后端，是routing hint，不是模型实际使用该卡计算的证据。负责显示窗口的WGPU适配器也可能与模型计算设备不同。
模型加载后的model_compute_context才是实际模型计算上下文的入口。查看其中的provider/backend、device/ordinal、GPU名称与架构、driver/API/runtime版本、已加载原生库路径、能力检查及当前选择；没有这项记录时不要仅凭显示清单宣称模型已在某张卡上运行。
CUDA还应核对compiled_cuda_profile：portable前端使用sm_80通用PTX，专用optimized前端为匹配的sm120a设备保留。两者都需要适用的基础GPU能力及支持CUDA13.1的驱动；专用前端不是对所有NVIDIA显卡的通用承诺。
AMD应以HIP实际枚举和选中的计算设备为准，不把DXGI显示顺序当作HIP序号；Windows RX7900XTX已经实测，其他设备仍需实际能力检查。
查看available/selected、disabled_reason/reason、fallback与错误链，区分“功能不可用”“没有被Auto选择”和“实际执行失败”。例如某项优化未通过数值检查、运行组件缺失、批量或区域不适用，都不等于显卡掉线。
Auto选择当前环境已验证可用的方案，可能保留普通执行；不要把其他显卡、其他系统或不同批量下的TPS套用到本机。程序不保证未经实测硬件的TPS。
普通兼容模式关闭可选加速，保留满足基础条件的GPU模型计算。它不会绕过基础GPU支持、驱动或必需运行组件。GPU基础路径不可用时，Auto会明确进入CPU兼容计算，界面和日志给出原因；CPU速度较慢。
缺少运行组件、网络下载失败或某个可选功能被禁用，不能据此判断为掉卡；不要据此自动重启主机、修改驱动或改变功耗。

三、可用命令

以下命令由同一个PrimitivePainter.exe提供，不需要Python。
若安装在默认目录，可以在PowerShell使用：

& "$env:LOCALAPPDATA\Programs\PrimitivePainter\PrimitivePainter.exe" --diagnose --report "$env:USERPROFILE\Desktop\PrimitivePainter-device-report.json"

--diagnose --report ABS
    只读查询显示/驱动及候选计算后端，写入指定绝对路径的报告；不读取模型权重、不执行模型推理，也不下载运行组件。

--backend auto|nvidia|amd|cpu
    指定自动选择、NVIDIA、AMD或CPU候选后端。默认auto。指定GPU不代表跳过真实支持条件检查。

--cuda-device N / --hip-device N
    指定对应计算运行时的非负设备序号，例如--backend amd --hip-device 0。序号不能直接从Windows显示适配器顺序推断。

--compatibility
    使用普通兼容执行设置；不绕过基础硬件与运行库要求。

--no-cpu-fallback
    调试用：GPU不可用时报告失败，停止自动回退到CPU。它不会强行启用未受支持的GPU功能。

--prepare-runtime --report ABS
    根据当前设备选择准备所需的固定版本原生运行组件，已有完整组件会复用；必要时联网下载。它不是只读命令，也不会强制同时准备NVIDIA与AMD两家组件。准备失败可记录原因并进入CPU兼容路径。

四、给其他AI或维护者的最小材料

提供应用版本、问题发生时间、执行了什么操作以及本会话诊断ZIP即可从事件链排查。
先沿launcher选择 → 运行组件准备/验证 → model_compute_context → 实际功能选择/禁用原因 → generation结束或error事件阅读。
错误链与耗时可解释失败位置和回退原因；没有实际生成计时就不要声称某个TPS已被验证。
本程序只在应用目录准备原生依赖，不安装显卡驱动、不修改系统PATH、不调整功耗。
'@
Write-Utf8 (Join-Path $payload 'DIAGNOSTICS.txt') ($diagnostics.Replace('@VERSION@',$Version))
$owned+=@('README.txt','THIRD_PARTY_NOTICES.txt','DIAGNOSTICS.txt')
$licenseLines=@()
foreach($entry in $sourceFiles.GetEnumerator()|Where-Object{$_.Key.StartsWith('licenses/')}){
    $relative=$entry.Key.Replace('/','\');$parent=Split-Path -Parent $relative
    $licenseLines+=('SetOutPath "$INSTDIR\'+$parent+'"')
    $licenseLines+=('File "${PAYLOAD}\'+$relative+'"')
}
$licenseLines+='SetOutPath "$INSTDIR"'
Write-Utf8 (Join-Path $stage 'install-licenses.nsh') ($licenseLines -join "`r`n")
Write-Utf8 (Join-Path $stage 'install-crt-root.nsh') ($crtRoot -join "`r`n")
Write-Utf8 (Join-Path $stage 'install-crt-runtime.nsh') ($crtRuntime -join "`r`n")
Write-Utf8 (Join-Path $stage 'install-crt-app.nsh') ($crtApp -join "`r`n")
$deletes=@($owned|Sort-Object -Unique|ForEach-Object {'Delete "$INSTDIR\'+$_.Replace('/','\')+'"'})
Write-Utf8 (Join-Path $stage 'uninstall-owned.nsh') ($deletes -join "`r`n")
Assert-SafeLiteral $OutputPath
New-Item -ItemType Directory -Path (Split-Path $OutputPath) -Force|Out-Null
$compiler=$NsisCompiler
if(!$compiler){$compiler=$env:NSIS_COMPILER}
if(!$compiler){$found=Get-Command makensis.exe -ErrorAction SilentlyContinue;if($found){$compiler=$found.Source}}
if(!$compiler){throw 'Install NSIS 3.x and supply -NsisCompiler, NSIS_COMPILER, or put makensis.exe on PATH. Compiler binaries are not included in this source release.'}
$compiler=[IO.Path]::GetFullPath($compiler)
Assert-File $compiler
$arguments=@('/V3','/INPUTCHARSET','UTF8',('/DVERSION='+$Version),('/DSTAGE='+$stage),('/DSETUP_FILE='+$OutputPath))
if($SyntaxCheck){$arguments+='/DSYNTAX_TEST'}
& $compiler @arguments (Join-Path $PSScriptRoot 'Universal.nsi')
if($LASTEXITCODE -ne 0){throw "NSIS universal installer compilation failed: $LASTEXITCODE"}
$result=[ordered]@{schema_version=1;version=$Version;syntax_test=[bool]$SyntaxCheck;stage=$stage;installer=$OutputPath;installer_bytes=(Get-Item -LiteralPath $OutputPath).Length;application_payload_files=$owned.Count;runtime_downloaded=$false;built_utc=[DateTime]::UtcNow.ToString('o')}
Write-Utf8 (Join-Path $stage 'build-result.json') ($result|ConvertTo-Json -Depth 5)
[pscustomobject]$result
