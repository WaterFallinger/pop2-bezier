param(
 [Parameter(Mandatory=$true)][string]$TorchRoot,
 [Parameter(Mandatory=$true)][string]$RocmCoreRoot,
 [Parameter(Mandatory=$true)][string]$HipSdkRoot,
 [string]$CompilerPath='', [string]$OutputDirectory=''
)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$app=[IO.Path]::GetFullPath((Join-Path $root '../..'))
if(!$OutputDirectory){$OutputDirectory=Join-Path $app 'target/aten-hip-bridge'}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
$TorchRoot=[IO.Path]::GetFullPath($TorchRoot);$RocmCoreRoot=[IO.Path]::GetFullPath($RocmCoreRoot);$HipSdkRoot=[IO.Path]::GetFullPath($HipSdkRoot)
foreach($path in @((Join-Path $TorchRoot 'include/ATen/ATen.h'),(Join-Path $RocmCoreRoot 'include/hip/hip_runtime.h'))){if(!(Test-Path -LiteralPath $path)){throw ('Missing development header: '+$path)}}
foreach($lib in @('torch.lib','torch_cpu.lib','torch_hip.lib','c10.lib','c10_hip.lib')){if(!(Test-Path -LiteralPath (Join-Path $TorchRoot ('lib/'+$lib)))){throw ('Missing ROCm LibTorch import library: '+$lib)}}
if(!(Test-Path -LiteralPath (Join-Path $HipSdkRoot 'lib/amdhip64.lib'))){throw 'The selected Windows HIP SDK does not contain lib/amdhip64.lib.'}
if(!$CompilerPath){$CompilerPath=Join-Path $HipSdkRoot 'bin/clang-cl.exe'}
if(!(Test-Path -LiteralPath $CompilerPath)){throw 'Supply the Windows HIP SDK clang-cl.exe through -CompilerPath or -HipSdkRoot.'}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if(!(Test-Path -LiteralPath $vswhere)){throw 'MSVC Build Tools with Windows SDK are required.'}
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(!$vs){throw 'MSVC x64 build tools are not installed.'}
$devcmd=Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
$lines=& cmd.exe /d /s /c ('call "'+$devcmd+'" -no_logo -arch=x64 -host_arch=x64 >nul && set')
if($LASTEXITCODE-ne0){throw 'MSVC environment preparation failed.'}
foreach($line in $lines){if($line-match'^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
New-Item -ItemType Directory -Path $OutputDirectory -Force|Out-Null
$production=@('checkpoint_loader.cpp','checkpoint_loader.h','kv_int8.cpp','kv_int8.h','painter_model.cpp','painter_model.h','painter_engine.cpp','painter_engine.h','painter_ffi.cpp','painter_ffi.h')
$memory=[IO.MemoryStream]::new()
foreach($name in $production){$path=Join-Path $root ('cpp/'+$name);$data=[Text.Encoding]::UTF8.GetBytes($name+[char]0);$memory.Write($data,0,$data.Length);$data=[IO.File]::ReadAllBytes($path);$memory.Write($data,0,$data.Length)}
$sha=[Security.Cryptography.SHA256]::Create();$buildId=([BitConverter]::ToString($sha.ComputeHash($memory.ToArray()))).Replace('-','').ToLowerInvariant();$sha.Dispose();$memory.Dispose()
[IO.File]::WriteAllText((Join-Path $OutputDirectory 'build_id.h'),('#pragma once'+[Environment]::NewLine+'#define PP_ATEN_BUILD_ID "'+$buildId+'"'+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
$sources=@('checkpoint_loader.cpp','kv_int8.cpp','painter_model.cpp','painter_engine.cpp','painter_ffi.cpp')|ForEach-Object{Join-Path $root ('cpp/'+$_)}
$output=Join-Path $OutputDirectory 'painter_aten_hip.dll'
$arguments=@('/nologo','/O2','/MD','/EHsc','/bigobj','/std:c++17','/Zc:__cplusplus','/utf-8','/DNOMINMAX','/LD','/DPP_ATEN_EXPORTS','/D__HIP_PLATFORM_AMD__=1','/DUSE_ROCM','/DTORCH_HIP_VERSION=701','/D__HIP_NO_HALF_OPERATORS__=1','/D__HIP_NO_HALF_CONVERSIONS__=1',('/I'+(Join-Path $TorchRoot 'include')),('/I'+(Join-Path $TorchRoot 'include/torch/csrc/api/include')),('/I'+(Join-Path $RocmCoreRoot 'include')),('/I'+$OutputDirectory))
$arguments+=$sources
$arguments+=@(('/Fe:'+$output),'/link',('/LIBPATH:'+(Join-Path $TorchRoot 'lib')),('/LIBPATH:'+(Join-Path $HipSdkRoot 'lib')),'torch.lib','torch_cpu.lib','torch_hip.lib','c10.lib','c10_hip.lib','amdhip64.lib')
Push-Location -LiteralPath $OutputDirectory
try{& $CompilerPath @arguments;if($LASTEXITCODE-ne0){throw ('ATen/HIP bridge build failed: '+$LASTEXITCODE)}}finally{Pop-Location}
[IO.File]::WriteAllText((Join-Path $OutputDirectory 'build-result.json'),(@{status='built';build_id=$buildId;library=$output;gpu_executed=$false}|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
Write-Output $output
