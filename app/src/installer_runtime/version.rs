use anyhow::{bail,Result};
use std::path::Path;

#[cfg(windows)]
pub fn file_version(path:&Path)->Result<[u16;4]>{
    use std::{ffi::c_void,os::windows::ffi::OsStrExt};
    #[link(name="version")]
    extern "system"{
        fn GetFileVersionInfoSizeW(name:*const u16,handle:*mut u32)->u32;
        fn GetFileVersionInfoW(name:*const u16,handle:u32,length:u32,data:*mut c_void)->i32;
        fn VerQueryValueW(block:*const c_void,subblock:*const u16,buffer:*mut *mut c_void,length:*mut u32)->i32;
    }
    #[repr(C)]
    struct FixedInfo{signature:u32,structure:u32,file_ms:u32,file_ls:u32,product_ms:u32,product_ls:u32,mask:u32,flags:u32,os:u32,kind:u32,subtype:u32,date_ms:u32,date_ls:u32}
    let name:Vec<u16>=path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut ignored=0;
    let length=unsafe{GetFileVersionInfoSizeW(name.as_ptr(),&mut ignored)};
    if length==0{bail!("{} 没有可读的 Windows FileVersion",path.display());}
    let mut data=vec![0u8;length as usize];
    if unsafe{GetFileVersionInfoW(name.as_ptr(),0,length,data.as_mut_ptr().cast())}==0{bail!("无法读取 {} 的版本资源：{}",path.display(),std::io::Error::last_os_error());}
    let root=[b'\\' as u16,0];let mut result=std::ptr::null_mut();let mut size=0;
    if unsafe{VerQueryValueW(data.as_ptr().cast(),root.as_ptr(),&mut result,&mut size)}==0 || result.is_null() || size<std::mem::size_of::<FixedInfo>() as u32{bail!("{} 的版本资源不完整",path.display());}
    // Version resource alignment is not promised by Vec<u8>.
    let info=unsafe{std::ptr::read_unaligned(result.cast::<FixedInfo>())};
    if info.signature!=0xfeef04bd{bail!("{} 的版本资源签名不正确",path.display());}
    Ok([(info.file_ms>>16) as u16,info.file_ms as u16,(info.file_ls>>16) as u16,info.file_ls as u16])
}
#[cfg(not(windows))]
pub fn file_version(_path:&Path)->Result<[u16;4]>{bail!("运行库版本核验仅支持 Windows")}
