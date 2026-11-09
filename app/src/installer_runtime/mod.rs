//! Version-pinned local CUDA runtime installation. No model/CUDA crate linkage.
pub mod archive;
pub mod download;
pub mod version;
pub mod running;

use anyhow::{bail,Context,Result};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::{collections::HashSet,io::{Read,Write},path::{Path,PathBuf}};

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Manifest{pub schema_version:u32,#[serde(default)]pub cuda_release:Option<String>,pub packages:Vec<Package>}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Package{pub id:String,pub version:String,pub url:String,pub sha256:String,pub size:u64,pub files:Vec<ExpectedFile>}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ExpectedFile{pub name:String,pub version:Option<[u16;4]>,pub sha256:String,pub size:u64}
#[derive(Clone,Debug,Serialize)]
pub struct InstalledFile{pub name:String,pub package:String,pub package_version:String,pub version:Option<[u16;4]>,pub sha256:String,pub size:u64,pub source:String}
#[derive(Clone,Debug,Serialize)]
pub struct InstallRecord{pub schema_version:u32,pub cuda_release:Option<String>,pub manifest_sha256:String,pub verified_at_unix:u64,pub files:Vec<InstalledFile>}

pub fn parse_manifest(text:&str)->Result<Manifest>{
    let manifest:Manifest=serde_json::from_str(text).context("内置运行库清单格式错误")?;
    if manifest.schema_version!=1||manifest.packages.is_empty(){bail!("不支持的运行库清单版本或空包列表");}
    let mut names=HashSet::new();let mut packages=HashSet::new();
    for package in &manifest.packages{
        safe_component(&package.id)?;safe_component(&package.version)?;
        if !packages.insert(package.id.to_ascii_lowercase())||package.files.is_empty(){bail!("运行库包重复或文件列表为空");}
        if !package.url.starts_with("https://developer.download.nvidia.com/compute/cuda/redist/"){bail!("清单不是锁定的 NVIDIA redist 官方地址");}
        check_hash(&package.sha256)?;if package.size==0{bail!("运行库包大小不能为零");}
        for file in &package.files{
            safe_component(&file.name)?;
            if !file.name.to_ascii_lowercase().ends_with(".dll")||!names.insert(file.name.to_ascii_lowercase()){bail!("运行库 DLL 白名单重复或含非 DLL 文件");}
            check_hash(&file.sha256)?;if file.size==0{bail!("运行库 DLL 大小不能为零");}
        }
    }
    Ok(manifest)
}
fn safe_component(value:&str)->Result<()>{
    if value.is_empty()||value=="."||value==".."||value.contains("..")||!value.bytes().all(|c|c.is_ascii_alphanumeric()||[b'-',b'_',b'.'].contains(&c)){bail!("不安全的运行库清单文件名：{value}");}Ok(())
}
fn check_hash(hash:&str)->Result<()>{if hash.len()!=64||!hash.bytes().all(|c|c.is_ascii_hexdigit()){bail!("清单 SHA256 格式不正确");}Ok(())}
fn hash_bytes(bytes:&[u8])->String{format!("{:x}",Sha256::digest(bytes))}
pub fn sha256_file(path:&Path)->Result<String>{
    let mut file=std::fs::File::open(path)?;let mut hasher=Sha256::new();let mut buffer=vec![0u8;4*1024*1024];
    loop{let length=file.read(&mut buffer)?;if length==0{break;}hasher.update(&buffer[..length]);}
    Ok(format!("{:x}",hasher.finalize()))
}
pub fn verify_bytes(path:&Path,size:u64,sha256:&str)->Result<()>{
    let metadata=std::fs::metadata(path).with_context(||format!("缺少文件 {}",path.display()))?;
    if !metadata.is_file()||metadata.len()!=size{bail!("{} 大小不匹配（实际 {}，预期 {size}）",path.display(),metadata.len());}
    let actual=sha256_file(path)?;if !actual.eq_ignore_ascii_case(sha256){bail!("{} SHA256 不匹配",path.display());}Ok(())
}
pub fn verify_file(path:&Path,file:&ExpectedFile)->Result<()>{
    verify_bytes(path,file.size,&file.sha256)?;
    if let Some(expected)=file.version{let actual=version::file_version(path)?;if actual!=expected{bail!("{} FileVersion {:?} 不匹配 {:?}",path.display(),actual,expected);}}
    Ok(())
}

fn record_file(package:&Package,file:&ExpectedFile,source:&str)->InstalledFile{
    InstalledFile{name:file.name.clone(),package:package.id.clone(),package_version:package.version.clone(),
        version:file.version,sha256:file.sha256.to_ascii_lowercase(),size:file.size,source:source.into()}
}
fn make_record(manifest:&Manifest,files:Vec<InstalledFile>)->Result<InstallRecord>{
    Ok(InstallRecord{schema_version:1,cuda_release:manifest.cuda_release.clone(),manifest_sha256:hash_bytes(&serde_json::to_vec(manifest)?),
        verified_at_unix:std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),files})
}
pub fn verify_runtime(manifest:&Manifest,runtime:&Path)->Result<InstallRecord>{
    let mut verified=Vec::new();
    for package in &manifest.packages{for file in &package.files{
        verify_file(&runtime.join(&file.name),file)?;verified.push(record_file(package,file,"verified_runtime"));
        println!("VERIFIED {}",file.name);
    }}
    make_record(manifest,verified)
}

pub fn install(manifest:&Manifest,runtime:&Path,cache:&Path,prefer_installed:bool)->Result<InstallRecord>{
    install_with_options(manifest,runtime,cache,prefer_installed,false)
}
pub fn install_with_options(manifest:&Manifest,runtime:&Path,cache:&Path,prefer_installed:bool,force_download:bool)->Result<InstallRecord>{
    let runtime=std::path::absolute(runtime)?;let cache=std::path::absolute(cache)?;
    std::fs::create_dir_all(&runtime)?;std::fs::create_dir_all(&cache)?;
    let toolkits=if prefer_installed&&!force_download{installed_toolkits()}else{Vec::new()};
    let mut verified=Vec::new();
    for package in &manifest.packages{
        println!("PACKAGE {} {}",package.id,package.version);
        let mut missing=Vec::new();
        for file in &package.files{
            let destination=runtime.join(&file.name);
            if !force_download&&verify_file(&destination,file).is_ok(){verified.push(record_file(package,file,"existing_runtime"));println!("REUSE_RUNTIME {}",file.name);continue;}
            let mut reused=false;
            for toolkit in &toolkits{
                for directory in [toolkit.join("bin/x64"),toolkit.join("bin")]{
                    let source=directory.join(&file.name);
                    if verify_file(&source,file).is_ok(){
                        println!("REUSE_INSTALLED {}",file.name);
                        copy_verified(&source,&destination,Some(file))?;
                        copy_installed_license(toolkit,&runtime,package)?;
                        verified.push(record_file(package,file,"installed_cuda_sha256_verified"));reused=true;break;
                    }
                }
                if reused{break;}
            }
            if !reused{missing.push(file);}
        }
        let name=format!("nvidia-{}-{}.zip",package.id,package.version);
        let zip=cache.join(&name);let part=cache.join(format!("{name}.part"));
        if missing.is_empty(){
            // Resume after a prior crash that finished DLL copies but had not
            // yet removed its known archive. Never leave duplicate payloads.
            if zip.exists(){remove_cache_file(&cache,&zip)?;}
            if part.exists(){remove_cache_file(&cache,&part)?;}
            println!("PACKAGE_COMPLETE {}",package.id);continue;
        }
        if force_download||verify_bytes(&zip,package.size,&package.sha256).is_err(){
            if zip.exists(){remove_cache_file(&cache,&zip)?;}
            println!("DOWNLOAD {} {}",package.id,package.url);
            download::download(&package.url,&part,package.size)?;
            verify_bytes(&part,package.size,&package.sha256).context("NVIDIA redist 包校验失败，未安装该包")?;
            replace_file(&part,&zip)?;
        }else{println!("REUSE_VERIFIED_ARCHIVE {}",package.id);}
        let extracted=archive::extract_package(&zip,&runtime,package)?;
        // Extraction validates all pinned package DLLs before any is committed.
        for file in &missing{
            let source=extracted.dll(&file.name).context("已验证的 ZIP 缺少预期 DLL")?;
            copy_verified(source,&runtime.join(&file.name),Some(file))?;
            verified.push(record_file(package,file,"nvidia_redist_sha256_verified"));
        }
        for (source,destination) in extracted.licenses(){copy_verified(source,destination,None)?;}
        // Drop cleans only files created in this helper's unique staging area.
        drop(extracted);
        remove_cache_file(&cache,&zip)?;
        if part.exists(){remove_cache_file(&cache,&part)?;}
        println!("PACKAGE_COMPLETE {}",package.id);
    }
    let record=make_record(manifest,verified)?;
    // Re-hash the final runtime, never trust only a saved install record.
    verify_runtime(manifest,&runtime)?;
    write_record(&runtime.join("runtime-install.json"),&record)?;
    Ok(record)
}

fn installed_toolkits()->Vec<PathBuf>{
    let program=std::env::var_os("ProgramFiles").map(PathBuf::from).unwrap_or_else(||PathBuf::from(r"C:\Program Files"));
    let root=program.join("NVIDIA GPU Computing Toolkit/CUDA");
    let mut paths=vec![root.join("v13.1")];
    if let Some(value)=std::env::var_os("CUDA_PATH"){paths.push(value.into());}
    let mut others:Vec<_>=std::fs::read_dir(&root).into_iter().flatten().filter_map(|entry|entry.ok())
        .filter(|v|v.path().is_dir()).map(|v|v.path()).collect();others.sort();others.reverse();paths.extend(others);
    let mut seen=HashSet::new();paths.retain(|p|p.is_dir()&&seen.insert(p.to_string_lossy().to_ascii_lowercase()));paths
}
fn copy_installed_license(toolkit:&Path,runtime:&Path,package:&Package)->Result<()>{
    for name in ["EULA.txt","LICENSE","LICENSE.txt"]{
        let source=toolkit.join(name);if source.is_file(){copy_verified(&source,&runtime.join("licenses").join(&package.id).join("LICENSE-installed.txt"),None)?;break;}
    }Ok(())
}
pub fn copy_verified(source:&Path,destination:&Path,expected:Option<&ExpectedFile>)->Result<()>{
    let parent=destination.parent().context("运行库目标缺少目录")?;std::fs::create_dir_all(parent)?;
    let name=destination.file_name().context("运行库目标缺少文件名")?.to_string_lossy();
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let temporary=parent.join(format!(".{name}-{}-{nonce}.installing",std::process::id()));
    let result=(||->Result<()>{
        let mut source=std::fs::File::open(source)?;let mut output=std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
        std::io::copy(&mut source,&mut output)?;output.sync_all()?;drop(output);
        if let Some(expected)=expected{verify_file(&temporary,expected)?;}
        replace_file(&temporary,destination)?;Ok(())
    })();
    if result.is_err(){let _=std::fs::remove_file(&temporary);}result
}
fn remove_cache_file(cache:&Path,path:&Path)->Result<()>{
    if path.parent()!=Some(cache)||!path.file_name().context("缓存文件名缺失")?.to_string_lossy().starts_with("nvidia-"){bail!("拒绝删除非本 helper 缓存路径");}
    let metadata=std::fs::symlink_metadata(path)?;if metadata.file_type().is_symlink()||!metadata.is_file(){bail!("拒绝删除非普通缓存文件");}
    std::fs::remove_file(path)?;Ok(())
}
fn write_record(path:&Path,record:&InstallRecord)->Result<()>{
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let temporary=path.with_extension(format!("json-{}-{nonce}.tmp",std::process::id()));
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(record)?)?;file.write_all(b"\n")?;file.sync_all()?;drop(file);
    let result=replace_file(&temporary,path);if result.is_err(){let _=std::fs::remove_file(&temporary);}result
}
#[cfg(windows)]
pub fn replace_file(source:&Path,destination:&Path)->Result<()>{
    use std::os::windows::ffi::OsStrExt;
    #[link(name="kernel32")]extern "system"{fn MoveFileExW(existing:*const u16,new:*const u16,flags:u32)->i32;}
    let source:Vec<u16>=source.as_os_str().encode_wide().chain(Some(0)).collect();let destination:Vec<u16>=destination.as_os_str().encode_wide().chain(Some(0)).collect();
    if unsafe{MoveFileExW(source.as_ptr(),destination.as_ptr(),0x1|0x8)}==0{bail!("无法更新安装文件（应用是否正在运行？）：{}",std::io::Error::last_os_error());}Ok(())
}
#[cfg(not(windows))]
pub fn replace_file(source:&Path,destination:&Path)->Result<()>{std::fs::rename(source,destination)?;Ok(())}
