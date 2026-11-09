//! Unicode Win32 file pickers. Cancellation never becomes a file operation.
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileKind { Project, Png, Svg, Image }

impl FileKind {
    pub(crate) fn extension(self) -> &'static str {
        match self { Self::Project=>"pproj",Self::Png|Self::Image=>"png",Self::Svg=>"svg" }
    }
    fn default_name(self) -> &'static str {
        match self { Self::Project=>"作品.pproj",Self::Png=>"作品.png",Self::Svg=>"作品.svg",Self::Image=>"参考.png" }
    }
    fn title(self, save:bool) -> &'static str {
        match (self,save) { (Self::Project,true)=>"保存状态树",(Self::Project,false)=>"导入状态树",
            (Self::Png,_)=>"导出 PNG",(Self::Svg,_)=>"导出 SVG",(Self::Image,_)=>"选择参考图像" }
    }
    fn filter(self) -> &'static str {
        match self { Self::Project=>"Primitive Painter 项目 (*.pproj)\0*.pproj\0\0",
            Self::Png=>"PNG 图像 (*.png)\0*.png\0\0",Self::Svg=>"SVG 矢量图 (*.svg)\0*.svg\0\0",
            Self::Image=>"参考图像 (*.png;*.jpg;*.jpeg;*.bmp;*.webp)\0*.png;*.jpg;*.jpeg;*.bmp;*.webp\0\0" }
    }
}

pub(crate) fn choose_save(kind:FileKind, initial:Option<&Path>) -> Result<Option<PathBuf>,String> {
    let suggested=initial.map(|path|path.with_extension(kind.extension()));
    native_dialog(kind,true,suggested.as_deref())
}
pub(crate) fn choose_import(initial:Option<&Path>) -> Result<Option<PathBuf>,String> {
    native_dialog(FileKind::Project,false,initial)
}
pub(crate) fn choose_reference_image(initial:Option<&Path>)->Result<Option<PathBuf>,String>{
    native_dialog(FileKind::Image,false,initial)
}

fn checked_selection(path:PathBuf,kind:FileKind)->Result<PathBuf,String>{
    // Never rewrite a selected extension after the OS overwrite prompt: doing
    // so could overwrite a different, unconfirmed destination.
    if kind==FileKind::Image{
        if !path.extension().and_then(|s|s.to_str()).is_some_and(|s|["png","jpg","jpeg","bmp","webp"].iter().any(|v|s.eq_ignore_ascii_case(v))){
            return Err("请选择 PNG、JPEG、BMP 或 WebP 图像。".into());
        }
        return Ok(path);
    }
    if !path.extension().and_then(|s|s.to_str()).is_some_and(|s|s.eq_ignore_ascii_case(kind.extension())) {
        return Err(format!("文件名需要以 .{} 结尾，请重新选择。",kind.extension()));
    }
    Ok(path)
}

#[cfg(target_os="windows")]
fn native_dialog(kind:FileKind,save:bool,initial:Option<&Path>)->Result<Option<PathBuf>,String>{
    use std::{ffi::{c_void,OsString},os::windows::ffi::{OsStrExt,OsStringExt},ptr};
    #[repr(C)]
    struct OpenFileNameW {
        size:u32,owner:*mut c_void,instance:*mut c_void,filter:*const u16,
        custom_filter:*mut u16,max_custom_filter:u32,filter_index:u32,file:*mut u16,
        max_file:u32,file_title:*mut u16,max_file_title:u32,initial_dir:*const u16,
        title:*const u16,flags:u32,file_offset:u16,file_extension:u16,default_extension:*const u16,
        custom_data:isize,hook:Option<unsafe extern "system" fn(*mut c_void,u32,usize,isize)->usize>,
        template_name:*const u16,reserved:*mut c_void,reserved_word:u32,flags_ex:u32,
    }
    #[link(name="comdlg32")]
    extern "system" {
        fn GetOpenFileNameW(data:*mut OpenFileNameW)->i32;
        fn GetSaveFileNameW(data:*mut OpenFileNameW)->i32;
        fn CommDlgExtendedError()->u32;
    }
    #[link(name="user32")]
    extern "system" { fn GetForegroundWindow()->*mut c_void; }
    let wide=|s:&str|s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<_>>();
    let title=wide(kind.title(save));let extension=wide(kind.extension());
    let filter=kind.filter().encode_utf16().collect::<Vec<_>>();
    let directory=initial.and_then(Path::parent).filter(|p|p.is_dir()).map(|p|
        p.as_os_str().encode_wide().chain(std::iter::once(0)).collect::<Vec<_>>());
    let name=initial.and_then(Path::file_name).map(|p|p.encode_wide().collect::<Vec<_>>())
        .unwrap_or_else(||if save{kind.default_name().encode_utf16().collect()}else{Vec::new()});
    if name.len()>=32768||name.contains(&0){return Err("文件名无效或过长。".into());}
    let mut file=vec![0u16;32768];file[..name.len()].copy_from_slice(&name);
    let mut data=OpenFileNameW {
        size:std::mem::size_of::<OpenFileNameW>()as u32,owner:unsafe{GetForegroundWindow()},instance:ptr::null_mut(),
        filter:filter.as_ptr(),custom_filter:ptr::null_mut(),max_custom_filter:0,filter_index:1,file:file.as_mut_ptr(),
        max_file:file.len()as u32,file_title:ptr::null_mut(),max_file_title:0,
        initial_dir:directory.as_ref().map_or(ptr::null(),|p|p.as_ptr()),title:title.as_ptr(),
        // Explorer + sizing + no cwd changes + existing parent. Save's native
        // overwrite prompt applies to the exact path returned to the caller.
        flags:0x0008_0000|0x0080_0000|0x0000_0008|0x0000_0800|if save{0x2}else{0x1000},
        file_offset:0,file_extension:0,default_extension:extension.as_ptr(),custom_data:0,hook:None,
        template_name:ptr::null(),reserved:ptr::null_mut(),reserved_word:0,flags_ex:0,
    };
    let selected=unsafe{if save{GetSaveFileNameW(&mut data)}else{GetOpenFileNameW(&mut data)}};
    if selected==0 {let error=unsafe{CommDlgExtendedError()};return if error==0{Ok(None)}else{Err(format!("无法打开文件对话框（错误码 0x{error:08X}）。"))};}
    let length=file.iter().position(|&c|c==0).ok_or("文件对话框返回了无效路径。")?;
    if length==0{return Ok(None);}
    checked_selection(PathBuf::from(OsString::from_wide(&file[..length])),kind).map(Some)
}

#[cfg(not(target_os="windows"))]
fn native_dialog(_:FileKind,_:bool,_:Option<&Path>)->Result<Option<PathBuf>,String>{Err("此文件对话框仅支持 Windows。".into())}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn unicode_paths_and_case_insensitive_extensions_are_preserved(){
        for(kind,name)in[(FileKind::Project,"人物 与分支.PPROJ"),(FileKind::Png,"图像 测试.PNG"),(FileKind::Svg,"矢量图.svg")]{
            let path=PathBuf::from(name);assert_eq!(checked_selection(path.clone(),kind).unwrap(),path);
        }
    }
    #[test]fn wrong_extensions_are_rejected_without_silently_changing_the_confirmed_target(){
        assert!(checked_selection(PathBuf::from("existing.png"),FileKind::Project).is_err());
        assert!(checked_selection(PathBuf::from("no-extension"),FileKind::Png).is_err());
    }
    #[test]fn reference_picker_accepts_common_raster_formats_without_changing_names(){
        for name in ["参考.PNG","脸部.JpG","样本.jpeg","旧图.BMP","图像.webp"]{
            let path=PathBuf::from(name);assert_eq!(checked_selection(path.clone(),FileKind::Image).unwrap(),path);
        }
        assert!(checked_selection(PathBuf::from("图像.svg"),FileKind::Image).is_err());
    }
    #[test]fn native_filters_are_double_terminated_and_each_kind_has_its_own_extension(){
        for kind in [FileKind::Project,FileKind::Png,FileKind::Svg]{
            assert!(kind.filter().ends_with("\0\0"));assert!(kind.filter().contains(&format!("*.{}",kind.extension())));
            assert!(kind.default_name().ends_with(kind.extension()));
        }
    }
}
