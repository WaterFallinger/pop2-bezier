//! Whitelist-only ZIP extraction. All package DLLs are staged/verified first.
use super::{verify_file,ExpectedFile,Package};
use anyhow::{bail,Context,Result};
use std::{collections::{HashMap,HashSet},io::{Read,Write},path::{Path,PathBuf}};

pub struct Extraction{root:PathBuf,created:Vec<PathBuf>,dlls:HashMap<String,PathBuf>,licenses:Vec<(PathBuf,PathBuf)>}
impl Extraction{
    pub fn dll(&self,name:&str)->Option<&Path>{self.dlls.get(&name.to_ascii_lowercase()).map(PathBuf::as_path)}
    pub fn licenses(&self)->impl Iterator<Item=(&Path,&Path)>{self.licenses.iter().map(|(a,b)|(a.as_path(),b.as_path()))}
}
impl Drop for Extraction{
    fn drop(&mut self){
        // Delete only individual files created in this unique stage, then
        // empty directories. Never recursively remove a user-supplied root.
        let mut dirs=HashSet::new();
        for path in &self.created{
            if !path.starts_with(&self.root){continue;}
            let _=std::fs::remove_file(path);
            let mut parent=path.parent();
            while let Some(dir)=parent{if !dir.starts_with(&self.root){break;}dirs.insert(dir.to_path_buf());if dir==self.root{break;}parent=dir.parent();}
        }
        dirs.insert(self.root.clone());let mut dirs:Vec<_>=dirs.into_iter().collect();
        dirs.sort_by_key(|p|std::cmp::Reverse(p.components().count()));
        for dir in dirs{let _=std::fs::remove_dir(dir);}
    }
}

pub fn extract_package(path:&Path,runtime:&Path,package:&Package)->Result<Extraction>{
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let root=runtime.join(format!(".runtime-stage-{}-{}-{nonce}",package.id,std::process::id()));
    std::fs::create_dir(&root)?;
    let mut output=Extraction{root:root.clone(),created:Vec::new(),dlls:HashMap::new(),licenses:Vec::new()};
    let requested:HashMap<String,&ExpectedFile>=package.files.iter().map(|f|(f.name.to_ascii_lowercase(),f)).collect();
    let mut names=HashSet::new();let mut targets=HashSet::new();
    let mut archive=zip::ZipArchive::new(std::fs::File::open(path)?).context("无法打开 NVIDIA redist ZIP")?;
    for index in 0..archive.len(){
        let mut entry=archive.by_index(index)?;
        let relative=safe_archive_path(entry.name())?;
        if entry.unix_mode().map(|m|m&0o170000==0o120000).unwrap_or(false){bail!("拒绝 ZIP 中的符号链接");}
        if entry.is_dir(){continue;}
        let normalized=relative.to_string_lossy().to_ascii_lowercase();
        if !names.insert(normalized){bail!("ZIP 中出现重复文件名");}
        let basename=relative.file_name().context("ZIP 文件名缺失")?.to_string_lossy().to_ascii_lowercase();
        let expected=requested.get(&basename).copied();
        let license=is_license(&basename);
        if expected.is_none()&&!license{continue;}
        let (staged,destination,limit)=if let Some(expected)=expected{
            if output.dlls.contains_key(&basename){bail!("ZIP 中出现同名 DLL 的多个版本：{basename}");}
            if entry.size()!=expected.size{bail!("ZIP 中 {} 大小与锁定清单不符",expected.name);}
            (root.join(&expected.name),runtime.join(&expected.name),expected.size)
        }else{
            if entry.size()>16*1024*1024{bail!("ZIP LICENSE 异常过大");}
            (root.join("licenses").join(&package.id).join(&relative),runtime.join("licenses").join(&package.id).join(&relative),16*1024*1024)
        };
        if !targets.insert(destination.to_string_lossy().to_ascii_lowercase()){bail!("ZIP 提取目标发生同名冲突");}
        std::fs::create_dir_all(staged.parent().context("ZIP 提取目标缺少父目录")?)?;
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&staged)?;
        output.created.push(staged.clone());
        let copied=std::io::copy(&mut entry.by_ref().take(limit+1),&mut file)?;
        if copied>limit{bail!("ZIP 内容超过白名单文件大小限制");}
        file.flush()?;file.sync_all()?;drop(file);
        if let Some(expected)=expected{
            verify_file(&staged,expected).with_context(||format!("ZIP 中 {} 内容校验失败",expected.name))?;
            output.dlls.insert(basename,staged);
        }else{output.licenses.push((staged,destination));}
    }
    for file in &package.files{if !output.dlls.contains_key(&file.name.to_ascii_lowercase()){bail!("ZIP 缺少白名单 DLL：{}",file.name);}}
    if output.licenses.is_empty(){bail!("NVIDIA redist ZIP 缺少 LICENSE/EULA，拒绝不带许可证安装");}
    Ok(output)
}

fn safe_archive_path(name:&str)->Result<PathBuf>{
    let normalized=name.replace('\\',"/");
    if normalized.starts_with('/')||normalized.contains(':')||normalized.contains('\0'){bail!("拒绝 ZIP 绝对路径或不安全路径");}
    let mut result=PathBuf::new();
    for part in normalized.split('/'){
        if part==".."{bail!("拒绝 ZIP 路径穿越");}
        if part.is_empty()||part=="."{continue;}
        // Windows trims final spaces/dots and recognizes device names.
        if part.ends_with(' ')||part.ends_with('.') {bail!("拒绝 ZIP 中含 Windows 路径歧义的名称");}
        let stem=part.split('.').next().unwrap_or("").to_ascii_uppercase();
        if ["CON","PRN","AUX","NUL"].contains(&stem.as_str())
            || ((stem.starts_with("COM")||stem.starts_with("LPT"))&&stem.len()==4&&stem.as_bytes()[3].is_ascii_digit()){
            bail!("拒绝 ZIP Windows 设备名称");
        }
        result.push(part);
    }
    if result.as_os_str().is_empty(){bail!("ZIP 路径为空");}Ok(result)
}
fn is_license(name:&str)->bool{
    ["license","eula","license.txt","license.md","license.rst","eula.txt","eula.md"].contains(&name)
        || (name.starts_with("license_")&&name.ends_with(".txt"))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn zip_paths_cannot_escape_or_alias_windows_files(){
        for name in ["../evil.dll","a/../../b","C:/Windows/x","/tmp/x",r"a\..\b","a/CON.txt","a/b.","a/file:stream"]{
            assert!(safe_archive_path(name).is_err(),"{name}");
        }
        assert_eq!(safe_archive_path("cuda/lib/x64/example.dll").unwrap(),PathBuf::from("cuda/lib/x64/example.dll"));
    }
    #[test]fn only_license_text_names_are_extra_outputs(){
        assert!(is_license("license.txt")&&is_license("eula.txt")&&is_license("license"));
        assert!(!is_license("readme.txt")&&!is_license("setup.exe")&&!is_license("license.exe"));
    }
}
