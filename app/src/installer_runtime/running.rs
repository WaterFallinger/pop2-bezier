use anyhow::Result;
use serde::Serialize;
use std::path::{Path,PathBuf};

#[derive(Debug,Serialize)]
pub struct RunningProcess{pub pid:u32,pub executable:PathBuf}

#[cfg(windows)]
pub fn check_running(app_dir:&Path)->Result<Vec<RunningProcess>>{
    use std::{ffi::c_void,os::windows::ffi::OsStringExt};
    type Handle=*mut c_void;
    #[repr(C)]
    struct Entry{size:u32,usage:u32,pid:u32,heap:usize,module:u32,threads:u32,parent:u32,priority:i32,flags:u32,name:[u16;260]}
    #[link(name="kernel32")]
    extern "system"{
        fn CreateToolhelp32Snapshot(flags:u32,pid:u32)->Handle;
        fn Process32FirstW(snapshot:Handle,entry:*mut Entry)->i32;
        fn Process32NextW(snapshot:Handle,entry:*mut Entry)->i32;
        fn OpenProcess(access:u32,inherit:i32,pid:u32)->Handle;
        fn QueryFullProcessImageNameW(process:Handle,flags:u32,name:*mut u16,size:*mut u32)->i32;
        fn CloseHandle(handle:Handle)->i32;
    }
    struct Owned(Handle);impl Drop for Owned{fn drop(&mut self){if !self.0.is_null(){unsafe{CloseHandle(self.0);}}}}
    let raw=unsafe{CreateToolhelp32Snapshot(2,0)};
    if raw==(-1isize as Handle)||raw.is_null(){anyhow::bail!("无法查询运行中的程序：{}",std::io::Error::last_os_error());}
    let snapshot=Owned(raw);
    let target=std::path::absolute(app_dir)?;
    let mut entry:Entry=unsafe{std::mem::zeroed()};entry.size=std::mem::size_of::<Entry>() as u32;
    let mut found=Vec::new();let mut exists=unsafe{Process32FirstW(snapshot.0,&mut entry)}!=0;
    while exists{
        let length=entry.name.iter().position(|&c|c==0).unwrap_or(entry.name.len());
        let name=String::from_utf16_lossy(&entry.name[..length]).to_ascii_lowercase();
        if is_application_name(&name){
            let handle=unsafe{OpenProcess(0x1000,0,entry.pid)};
            if !handle.is_null(){
                let process=Owned(handle);let mut buffer=vec![0u16;32768];let mut size=buffer.len() as u32;
                if unsafe{QueryFullProcessImageNameW(process.0,0,buffer.as_mut_ptr(),&mut size)}!=0{
                    let path=PathBuf::from(std::ffi::OsString::from_wide(&buffer[..size as usize]));
                    if belongs_to_installation(&target,&path){found.push(RunningProcess{pid:entry.pid,executable:path});}
                }
            }
        }
        exists=unsafe{Process32NextW(snapshot.0,&mut entry)}!=0;
    }
    Ok(found)
}

fn is_versioned_application(name:&str)->bool{
    let Some(version)=name.strip_prefix("primitive-painter-").and_then(|v|v.strip_suffix(".exe")) else{return false;};
    let parts=version.split('.').collect::<Vec<_>>();
    parts.len()==3 && parts.iter().all(|p|!p.is_empty() && p.bytes().all(|c|c.is_ascii_digit()))
}

fn is_application_name(name:&str)->bool{
    name=="primitivepainter.exe"||name=="primitive-painter.exe"||is_versioned_application(name)||is_unified_frontend(name)
}

fn is_unified_frontend(name:&str)->bool{
    matches!(name,"primitive-painter-cuda-optimized.exe"|"primitive-painter-cuda-portable.exe"|"primitive-painter-aten-hip.exe"|"primitive-painter-cpu.exe")
}
fn is_app_directory(path:&Path)->bool{
    let Some(name)=path.file_name().and_then(|s|s.to_str()) else{return false};
    let name=name.to_ascii_lowercase();
    let Some(version)=name.strip_prefix("app-") else{return false};
    let parts=version.split('.').collect::<Vec<_>>();
    parts.len()==3&&parts.iter().all(|p|!p.is_empty()&&p.bytes().all(|b|b.is_ascii_digit()))
        &&path.parent().and_then(|p|p.file_name()).is_some_and(|n|n.eq_ignore_ascii_case("runtime"))
}

fn belongs_to_installation(target:&Path,path:&Path)->bool{
    let Some(name)=path.file_name().and_then(|s|s.to_str()) else{return false;};
    let name=name.to_ascii_lowercase();
    let Some(parent)=path.parent() else{return false;};
    let in_root=normalized(parent)==normalized(target);
    let in_runtime=normalized(parent)==normalized(&target.join("runtime"));
    (in_root && (name=="primitivepainter.exe"||name=="primitive-painter.exe"))
        || (in_runtime && (name=="primitive-painter.exe"||is_versioned_application(&name)))
        || (is_unified_frontend(&name)&&is_app_directory(parent)
            &&((in_root&&is_app_directory(target))
                ||parent.parent().is_some_and(|p|normalized(p)==normalized(&target.join("runtime")))))
        || (name=="primitivepainter.exe"&&is_app_directory(target)
            &&target.parent().and_then(|p|p.parent()).is_some_and(|p|normalized(p)==normalized(parent)))
}
#[cfg(not(windows))]
pub fn check_running(_app_dir:&Path)->Result<Vec<RunningProcess>>{anyhow::bail!("安装运行状态检查仅支持 Windows")}

fn normalized(path:&Path)->String{
    let absolute=std::fs::canonicalize(path).or_else(|_|std::path::absolute(path)).unwrap_or_else(|_|path.to_path_buf());
    let text=absolute.to_string_lossy();
    let text=if let Some(rest)=text.strip_prefix(r"\\?\UNC\"){format!(r"\\{rest}")}
        else{text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned()};
    text.replace('/',r"\").to_lowercase()
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn exact_paths_not_prefixes_are_compared(){
        let a=normalized(Path::new(r"C:\Apps\PrimitivePainter\runtime\primitive-painter.exe"));
        let b=normalized(Path::new(r"C:\Apps\PrimitivePainter2\runtime\primitive-painter.exe"));
        assert_ne!(a,b);
    }
    #[test]fn versioned_processes_are_scoped_to_exact_install_runtime(){
        let root=Path::new(r"C:\Apps\PrimitivePainter");
        assert!(belongs_to_installation(root,Path::new(r"C:\Apps\PrimitivePainter\runtime\primitive-painter-0.5.5.exe")));
        assert!(!belongs_to_installation(root,Path::new(r"C:\Apps\PrimitivePainter2\runtime\primitive-painter-0.5.5.exe")));
        assert!(!belongs_to_installation(root,Path::new(r"C:\Apps\PrimitivePainter\models\primitive-painter-0.5.5.exe")));
        for name in ["primitive-painter-backup.exe","primitive-painter-0..5.exe","primitive-painter-0.5.exe","primitive-painter-0.5.5.exe.old"]{
            assert!(!is_versioned_application(name));
        }
        assert!(is_versioned_application("primitive-painter-10.20.300.exe"));
    }
    #[test]fn unified_frontends_match_only_known_names_in_exact_version_directories(){
        let root=Path::new(r"C:\Apps\PrimitivePainter");
        let current=root.join(r"runtime\app-0.9.0");
        for name in ["primitive-painter-cuda-optimized.exe","primitive-painter-cuda-portable.exe","primitive-painter-aten-hip.exe","primitive-painter-cpu.exe"]{
            assert!(is_application_name(name));
            assert!(belongs_to_installation(root,&current.join(name)));
            assert!(belongs_to_installation(&current,&current.join(name)));
            assert!(!belongs_to_installation(&current,&root.join(r"runtime\app-0.8.0").join(name)));
            assert!(!belongs_to_installation(root,&root.join("models").join(name)));
            assert!(!belongs_to_installation(root,&root.join(r"runtime\app-latest").join(name)));
            assert!(!belongs_to_installation(root,&root.join(r"runtime\app-0.9.0\nested").join(name)));
        }
        assert!(!belongs_to_installation(root,&current.join("user-tool.exe")));
        assert!(belongs_to_installation(&current,&root.join("PrimitivePainter.exe")));
        assert!(!belongs_to_installation(root,Path::new(r"C:\Apps\PrimitivePainter2\runtime\app-0.9.0\primitive-painter-cpu.exe")));
    }
}
