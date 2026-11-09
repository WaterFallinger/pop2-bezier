//! HTTPS streaming with the operating system's certificate validation intact.
use anyhow::{bail,Context,Result};
use std::path::Path;

#[derive(Clone,Debug)]
struct HttpsUrl{host:String,path:String}
fn parse_url(url:&str)->Result<HttpsUrl>{
    let tail=url.strip_prefix("https://").context("运行库下载只允许 HTTPS")?;
    let split=tail.find('/').unwrap_or(tail.len());let host=&tail[..split];
    if host.is_empty() || host.chars().any(|c|matches!(c,'@'|':'|'\\'|'#'|'?')){bail!("运行库 URL 主机不规范");}
    let lower=host.to_ascii_lowercase();
    if !(lower.ends_with(".nvidia.com")||lower=="nvidia.com"||lower.ends_with(".nvidia.cn")||lower=="nvidia.cn"){bail!("拒绝非 NVIDIA 官方下载主机：{host}");}
    let path=if split==tail.len(){"/"}else{&tail[split..]};
    if path.chars().any(|c|matches!(c,'\0'|'\\'|'#')){bail!("运行库 URL 路径不规范");}
    Ok(HttpsUrl{host:host.into(),path:path.into()})
}
fn redirect_url(current:&str,location:&str)->Result<String>{
    let current=parse_url(current)?;
    let location=location.trim();
    if location.is_empty(){bail!("NVIDIA 重定向地址为空");}
    if location.contains(':')&&!location.starts_with("https://"){bail!("拒绝 HTTP 降级或非 HTTPS 重定向");}
    let next=if location.starts_with("https://"){location.to_owned()}
        else if location.starts_with("//"){format!("https:{location}")}
        else if location.starts_with('/'){format!("https://{}{}",current.host,location)}
        else{let parent=current.path.rsplit_once('/').map(|p|p.0).unwrap_or("");format!("https://{}{parent}/{location}",current.host)};
    parse_url(&next)?;Ok(next)
}

#[cfg(windows)]
pub fn download(url:&str,part:&Path,expected_size:u64)->Result<()>{
    use std::{ffi::c_void,io::Write};
    type Handle=*mut c_void;
    #[link(name="winhttp")]
    extern "system"{
        fn WinHttpOpen(agent:*const u16,access:u32,proxy:*const u16,bypass:*const u16,flags:u32)->Handle;
        fn WinHttpSetTimeouts(session:Handle,resolve:i32,connect:i32,send:i32,receive:i32)->i32;
        fn WinHttpConnect(session:Handle,server:*const u16,port:u16,reserved:u32)->Handle;
        fn WinHttpOpenRequest(connection:Handle,verb:*const u16,path:*const u16,version:*const u16,referer:*const u16,accept:*const *const u16,flags:u32)->Handle;
        fn WinHttpSetOption(handle:Handle,option:u32,value:*mut c_void,length:u32)->i32;
        fn WinHttpSendRequest(request:Handle,headers:*const u16,header_len:u32,optional:*mut c_void,optional_len:u32,total:u32,context:usize)->i32;
        fn WinHttpReceiveResponse(request:Handle,reserved:*mut c_void)->i32;
        fn WinHttpQueryHeaders(request:Handle,level:u32,name:*const u16,buffer:*mut c_void,length:*mut u32,index:*mut u32)->i32;
        fn WinHttpReadData(request:Handle,buffer:*mut c_void,length:u32,read:*mut u32)->i32;
        fn WinHttpCloseHandle(handle:Handle)->i32;
    }
    struct Owned(Handle);
    impl Drop for Owned{fn drop(&mut self){if !self.0.is_null(){unsafe{WinHttpCloseHandle(self.0);}}}}
    fn wide(value:&str)->Vec<u16>{value.encode_utf16().chain(Some(0)).collect()}
    fn checked(raw:Handle,what:&str)->Result<Owned>{if raw.is_null(){bail!("{what}：{}",std::io::Error::last_os_error());}Ok(Owned(raw))}
    fn header(request:Handle,query:u32)->Result<String>{
        let mut bytes=0;
        unsafe{WinHttpQueryHeaders(request,query,std::ptr::null(),std::ptr::null_mut(),&mut bytes,std::ptr::null_mut());}
        if bytes==0{bail!("HTTP 响应缺少重定向地址");}
        let mut buffer=vec![0u16;(bytes as usize+1)/2];
        if unsafe{WinHttpQueryHeaders(request,query,std::ptr::null(),buffer.as_mut_ptr().cast(),&mut bytes,std::ptr::null_mut())}==0{bail!("无法读取 HTTP 响应头：{}",std::io::Error::last_os_error());}
        let end=buffer.iter().position(|&v|v==0).unwrap_or(buffer.len());Ok(String::from_utf16(&buffer[..end])?)
    }
    let agent=wide("PrimitivePainter-RuntimeInstaller/1.0");
    let session=checked(unsafe{WinHttpOpen(agent.as_ptr(),4,std::ptr::null(),std::ptr::null(),0)},"无法建立系统 HTTPS 会话")?;
    if unsafe{WinHttpSetTimeouts(session.0,60_000,60_000,60_000,120_000)}==0{bail!("无法设置 HTTPS 超时：{}",std::io::Error::last_os_error());}
    let mut url=url.to_owned();
    for redirects in 0..=8{
        let parsed=parse_url(&url)?;let host=wide(&parsed.host);let path=wide(&parsed.path);let verb=wide("GET");
        let connection=checked(unsafe{WinHttpConnect(session.0,host.as_ptr(),443,0)},"无法连接 NVIDIA 下载服务器")?;
        let request=checked(unsafe{WinHttpOpenRequest(connection.0,verb.as_ptr(),path.as_ptr(),std::ptr::null(),std::ptr::null(),std::ptr::null(),0x0080_0000)},"无法创建 HTTPS 下载请求")?;
        // Handle redirects explicitly so every target remains HTTPS/NVIDIA.
        let mut policy=0u32;
        if unsafe{WinHttpSetOption(request.0,88,(&mut policy as *mut u32).cast(),4)}==0{bail!("无法设置 HTTPS 重定向策略");}
        if unsafe{WinHttpSendRequest(request.0,std::ptr::null(),0,std::ptr::null_mut(),0,0,0)}==0{bail!("HTTPS 请求失败：{}",std::io::Error::last_os_error());}
        if unsafe{WinHttpReceiveResponse(request.0,std::ptr::null_mut())}==0{bail!("HTTPS 响应失败：{}",std::io::Error::last_os_error());}
        let mut status=0u32;let mut length=4;
        if unsafe{WinHttpQueryHeaders(request.0,19|0x2000_0000,std::ptr::null(),(&mut status as *mut u32).cast(),&mut length,std::ptr::null_mut())}==0{bail!("无法读取 HTTP 状态码");}
        if [301,302,303,307,308].contains(&status){
            if redirects==8{bail!("NVIDIA 下载重定向次数过多");}
            url=redirect_url(&url,&header(request.0,33)?)?;continue;
        }
        if status!=200{bail!("NVIDIA 下载返回 HTTP {status}，仅接受完整 200 响应");}
        if let Ok(metadata)=std::fs::symlink_metadata(part){if metadata.file_type().is_symlink(){bail!("拒绝写入符号链接缓存文件");}}
        let mut file=std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(part)?;
        let mut buffer=vec![0u8;1024*1024];let mut total=0u64;let mut next_progress=16*1024*1024;
        loop{let mut read=0;
            if unsafe{WinHttpReadData(request.0,buffer.as_mut_ptr().cast(),buffer.len() as u32,&mut read)}==0{bail!("下载中断：{}",std::io::Error::last_os_error());}
            if read==0{break;}total+=read as u64;
            if total>expected_size{bail!("下载内容大于锁定包大小，已停止写入");}
            file.write_all(&buffer[..read as usize])?;
            if total>=next_progress{println!("DOWNLOAD_PROGRESS {total} {expected_size}");std::io::stdout().flush()?;next_progress=total+16*1024*1024;}
        }
        file.sync_all()?;
        if total!=expected_size{bail!("下载大小不符：实际 {total}，预期 {expected_size}");}
        println!("DOWNLOAD_PROGRESS {total} {expected_size}");std::io::stdout().flush()?;return Ok(());
    }
    bail!("NVIDIA 下载未完成")
}
#[cfg(not(windows))]
pub fn download(_url:&str,_part:&Path,_expected_size:u64)->Result<()>{bail!("原生 WinHTTP 下载器仅支持 Windows")}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn only_official_https_hosts_and_safe_redirects_are_accepted(){
        assert!(parse_url("https://developer.download.nvidia.com/compute/archive.zip").is_ok());
        assert!(parse_url("http://developer.download.nvidia.com/a").is_err());
        assert!(parse_url("https://evilnvidia.com/a").is_err());
        assert!(parse_url("https://nvidia.com@evil.com/a").is_err());
        assert!(redirect_url("https://developer.download.nvidia.com/a/b","http://evil.com/x").is_err());
        assert_eq!(redirect_url("https://developer.download.nvidia.com/a/b","/c").unwrap(),"https://developer.download.nvidia.com/c");
    }
}
