param(
 [switch]$CpuOnly,[switch]$ToolsOnly,[switch]$AmdOnly,
 [switch]$DebugBuild,[switch]$CheckOnly,[switch]$Test,[switch]$Offline,
 [string[]]$Bin=@(),[ValidateSet('optimized','portable')][string]$CudaProfile='portable',
 [string]$TargetDirectory='',[string]$CudaToolkitRoot='', [int]$Jobs=4
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if(@($CpuOnly,$ToolsOnly,$AmdOnly|Where-Object{$_}).Count-gt1){throw 'Choose only one of CpuOnly, ToolsOnly, AmdOnly.'}
if($Jobs-lt1){throw 'Jobs must be positive.'}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if(Test-Path -LiteralPath $vswhere){
 $vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
 if($vs){
  $devcmd=Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
  $lines=& cmd.exe /d /s /c ('call "'+$devcmd+'" -no_logo -arch=x64 -host_arch=x64 >nul && set')
  if($LASTEXITCODE-ne0){throw 'MSVC environment preparation failed.'}
  foreach($line in $lines){if($line-match'^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
 }
}
$cuda=!$CpuOnly-and!$ToolsOnly-and!$AmdOnly
if($CpuOnly){$manifest=Join-Path $root 'backends/cpu/Cargo.toml';$defaultBins=@('primitive-painter-cpu');$defaultTarget='target-cpu'}
elseif($AmdOnly){$manifest=Join-Path $root 'native/aten_hip/gui/Cargo.toml';$defaultBins=@('primitive-painter-aten-hip');$defaultTarget='target-aten'}
elseif($ToolsOnly){$manifest=Join-Path $root 'Cargo.toml';$defaultBins=@('PrimitivePainter','runtime-installer','aten-runtime-installer');$defaultTarget='target-tools'}
else{$manifest=Join-Path $root 'Cargo.toml';$defaultBins=@('primitive-painter');$defaultTarget='target-cuda-'+$CudaProfile}
if(!$Bin.Count){$Bin=$defaultBins}
if(!$TargetDirectory){$TargetDirectory=Join-Path $root $defaultTarget}
$env:CARGO_TARGET_DIR=[IO.Path]::GetFullPath($TargetDirectory)
$env:CARGO_BUILD_JOBS=[string]$Jobs
$env:PAINTER_CUDA_PROFILE=$CudaProfile
$env:PAINTER_CUDA_TOOLKIT_VERSION='none'
if($cuda){
 if(!$CudaToolkitRoot){$CudaToolkitRoot=$env:CUDA_PATH}
 if(!$CudaToolkitRoot){throw 'CUDA worker builds require CUDA Toolkit 13.1. Supply -CudaToolkitRoot or CUDA_PATH; end users do not need the toolkit.'}
 $nvcc=Join-Path $CudaToolkitRoot 'bin/nvcc.exe'
 if(!(Test-Path -LiteralPath $nvcc)){throw 'nvcc.exe not found under the supplied CUDA toolkit.'}
 $version=& $nvcc --version
 if($LASTEXITCODE-ne0-or($version-join' ')-notmatch'release 13\.1,'){throw 'This runtime manifest and PTX profiles require CUDA Toolkit 13.1.'}
 $env:CUDA_PATH=$CudaToolkitRoot;$env:CUDA_HOME=$CudaToolkitRoot;$env:CUDA_ROOT=$CudaToolkitRoot;$env:NVCC=$nvcc
 $env:PATH=(Join-Path $CudaToolkitRoot 'bin')+';'+$env:PATH
 $env:LIB=(Join-Path $CudaToolkitRoot 'lib/x64')+';'+$env:LIB
 $env:CUDARC_CUDA_VERSION='13010';$env:PAINTER_CUDA_TOOLKIT_VERSION='13.1'
 $env:CUDA_COMPUTE_CAP=if($CudaProfile-eq'portable'){'80'}else{'120'}
}
$action=if($Test){'test'}elseif($CheckOnly){'check'}else{'build'}
$arguments=@($action,'--locked','--manifest-path',$manifest)
if($Offline){$arguments+='--offline'}
if(!$DebugBuild-and!$CheckOnly){$arguments+='--release'}
if($cuda){$arguments+=@('--features','cuda')}
foreach($name in $Bin){$arguments+=@('--bin',$name)}
& cargo @arguments
if($LASTEXITCODE-ne0){throw ('Native build failed: '+$LASTEXITCODE)}
if($cuda){
 $configuration=if($DebugBuild-or$CheckOnly){'debug'}else{'release'}
 $buildRoot=Join-Path $env:CARGO_TARGET_DIR ($configuration+'/build')
 $expected=if($CudaProfile-eq'portable'){'sm_80'}else{'sm_120a'}
 $ptx=@()
 foreach($build in (Get-ChildItem -LiteralPath $buildRoot -Directory -Filter 'candle-kernels-*')){
  $out=Join-Path $build.FullName 'out';if(!(Test-Path -LiteralPath $out)){continue}
  foreach($file in (Get-ChildItem -LiteralPath $out -Recurse -File -Filter '*.ptx')){
   $target=Get-Content -LiteralPath $file.FullName -TotalCount 20|Where-Object{$_-match'^\.target\s+(\S+)'}|Select-Object -First 1
   if(!$target){throw 'Generated PTX does not contain a target header.'}
   $actual=[regex]::Match($target,'^\.target\s+(\S+)').Groups[1].Value.TrimEnd(',')
   if($actual-ne$expected){throw ('PTX profile mismatch: '+$actual+' expected '+$expected)}
   $ptx+=@{file=$file.Name;target=$actual}
  }
 }
 if(!$ptx.Count){throw 'No generated PTX was available to verify.'}
 $metadata=[ordered]@{schema_version=1;profile=$CudaProfile;embedded_ptx_target=$expected;minimum_compute_capability=if($CudaProfile-eq'portable'){'8.0'}else{'12.0 exact'};minimum_driver_api=13010;cuda_toolkit='13.1';dtype='BF16';optional_kernels='NVRTC validates the actual runtime device';architecture_specific=($CudaProfile-eq'optimized');embedded_ptx_verified=$true;ptx_count=$ptx.Count;action=$action;output_directory=$env:CARGO_TARGET_DIR;finished_utc=[DateTime]::UtcNow.ToString('o')}
 [IO.File]::WriteAllText((Join-Path $env:CARGO_TARGET_DIR 'cuda-build-profile.json'),($metadata|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
