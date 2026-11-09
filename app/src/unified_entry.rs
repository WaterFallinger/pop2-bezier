#![cfg_attr(windows, windows_subsystem = "windows")]
mod diagnostics;
mod hardware_inventory;
mod platform_check;
mod unified_launcher;

fn main(){
    if let Err(error)=unified_launcher::run(){
        diagnostics::error("launcher",&error);
        let details=format!("Primitive Painter 无法启动：{error:#}\n诊断日志：{}",
            diagnostics::directory().map(|p|p.display().to_string()).unwrap_or_else(||"日志目录不可写".into()));
        if std::env::args().any(|arg|matches!(arg.as_str(),"--diagnose"|"--prepare-runtime")){
            eprintln!("{details}");
        }else{show_error(&details);}
        diagnostics::flush_summary();
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn show_error(message:&str){
    #[link(name="user32")]
    extern "system"{fn MessageBoxW(window:*mut std::ffi::c_void,text:*const u16,title:*const u16,flags:u32)->i32;}
    let message=message.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    let title="Primitive Painter".encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    unsafe{MessageBoxW(std::ptr::null_mut(),message.as_ptr(),title.as_ptr(),0x10);}
}
#[cfg(not(windows))]
fn show_error(message:&str){eprintln!("{message}");}
