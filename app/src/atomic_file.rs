//! Replace a user-chosen output only after the complete new file is durable.
use std::{fs::{self,File,OpenOptions}, io::Write, path::{Path,PathBuf},
    sync::atomic::{AtomicU64,Ordering}};

struct PendingFile { path:PathBuf, file:Option<File>, committed:bool }
impl Drop for PendingFile {
    fn drop(&mut self) {
        self.file.take();
        if !self.committed { let _=fs::remove_file(&self.path); }
    }
}

pub(crate) fn write_atomic(
    path:&Path,
    write:impl FnOnce(&mut File)->Result<(),String>,
) -> Result<(),String> {
    static NEXT:AtomicU64=AtomicU64::new(1);
    let name=path.file_name().filter(|name|!name.is_empty()).ok_or("请选择有效的保存文件名")?;
    let parent=path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let parent=fs::canonicalize(parent).map_err(|e|format!("无法访问保存目录：{e}"))?;
    if !parent.is_dir(){return Err("保存目录不是文件夹".into());}
    let destination=parent.join(name);
    if destination.is_dir(){return Err("保存目标是文件夹，请选择文件名".into());}
    let (temporary,file)=(0..32).find_map(|_|{
        let id=NEXT.fetch_add(1,Ordering::Relaxed);
        let temporary=parent.join(format!(".primitive-painter-{}-{id}.tmp",std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&temporary){
            Ok(file)=>Some(Ok((temporary,file))),
            Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>None,
            Err(e)=>Some(Err(format!("无法创建保存临时文件：{e}"))),
        }
    }).ok_or("无法取得独立的保存临时文件")??;
    let mut pending=PendingFile{path:temporary,file:Some(file),committed:false};
    let file=pending.file.as_mut().ok_or("保存文件未打开")?;
    write(file)?;
    file.flush().and_then(|_|file.sync_all()).map_err(|e|format!("保存内容未能完整写入磁盘：{e}"))?;
    pending.file.take();
    replace(&pending.path,&destination)?;
    pending.committed=true;
    Ok(())
}

#[cfg(windows)]
fn replace(from:&Path,to:&Path)->Result<(),String>{
    use std::os::windows::ffi::OsStrExt;
    #[link(name="kernel32")]
    extern "system"{fn MoveFileExW(from:*const u16,to:*const u16,flags:u32)->i32;}
    let wide=|path:&Path|->Result<Vec<u16>,String>{
        let mut value=path.as_os_str().encode_wide().collect::<Vec<_>>();
        if value.contains(&0){return Err("文件路径包含无效字符".into());}
        value.push(0);Ok(value)
    };
    let from=wide(from)?;let to=wide(to)?;
    if unsafe{MoveFileExW(from.as_ptr(),to.as_ptr(),0x1|0x8)}==0{
        return Err(format!("无法替换目标文件，原文件已保留：{}",std::io::Error::last_os_error()));
    }
    Ok(())
}
#[cfg(not(windows))]
fn replace(from:&Path,to:&Path)->Result<(),String>{
    fs::rename(from,to).map_err(|e|format!("无法替换目标文件，原文件已保留：{e}"))
}

#[cfg(test)]
mod tests{
    use super::*;
    fn directory()->PathBuf{
        static NEXT:AtomicU64=AtomicU64::new(1);
        let path=std::env::temp_dir().join(format!("painter-atomic-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir_all(&path).unwrap();path
    }
    #[test]
    fn completed_outputs_replace_atomically_and_writer_failure_keeps_old_bytes(){
        let dir=directory();let path=dir.join("项目 测试.pproj");
        write_atomic(&path,|f|f.write_all(b"old").map_err(|e|e.to_string())).unwrap();
        assert!(write_atomic(&path,|f|{f.write_all(b"partial").unwrap();Err("interrupted".into())}).is_err());
        assert_eq!(fs::read(&path).unwrap(),b"old");
        write_atomic(&path,|f|f.write_all(b"complete").map_err(|e|e.to_string())).unwrap();
        assert_eq!(fs::read(&path).unwrap(),b"complete");
        assert_eq!(fs::read_dir(&dir).unwrap().count(),1);
        fs::remove_file(path).unwrap();fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn replacement_failure_and_panics_cleanup_owned_temporary_files(){
        let dir=directory();let path=dir.join("target");fs::create_dir(&path).unwrap();
        assert!(write_atomic(&path,|_|Ok(())).is_err());
        assert_eq!(fs::read_dir(&dir).unwrap().count(),1);
        let file=dir.join("panic.png");
        assert!(std::panic::catch_unwind(||write_atomic(&file,|_|panic!("test writer panic"))).is_err());
        assert!(!file.exists());assert_eq!(fs::read_dir(&dir).unwrap().count(),1);
        fs::remove_dir(path).unwrap();fs::remove_dir(dir).unwrap();
    }
}
