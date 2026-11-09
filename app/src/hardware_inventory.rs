//! Display inventory is a routing hint, never evidence of model compute support.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_bytes: u64,
    pub shared_bytes: u64,
    pub luid: String,
    pub software: bool,
}

#[cfg(windows)]
pub fn enumerate() -> Result<Vec<Adapter>, String> {
    use std::ffi::c_void;
    #[repr(C)] struct Guid { a:u32,b:u16,c:u16,d:[u8;8] }
    #[repr(C)] struct Desc {
        name:[u16;128],vendor:u32,device:u32,subsystem:u32,revision:u32,
        dedicated:usize,system:usize,shared:usize,luid_low:u32,luid_high:i32,flags:u32,
    }
    #[link(name="dxgi")]
    extern "system" { fn CreateDXGIFactory1(iid:*const Guid,out:*mut *mut c_void)->i32; }
    struct Com(*mut c_void);
    impl Com {
        unsafe fn slot(&self,index:usize)->*mut c_void { *(*(self.0 as *mut *mut *mut c_void)).add(index) }
    }
    impl Drop for Com { fn drop(&mut self){unsafe{
        let release:unsafe extern "system" fn(*mut c_void)->u32=std::mem::transmute(self.slot(2));release(self.0);
    }} }
    let iid=Guid{a:0x770aae78,b:0xf26f,c:0x4dba,d:[0xa8,0x29,0x25,0x3c,0x83,0xd1,0xb3,0x87]};
    let mut raw=std::ptr::null_mut();
    let hr=unsafe{CreateDXGIFactory1(&iid,&mut raw)};
    if hr<0||raw.is_null(){return Err(format!("DXGI inventory unavailable: 0x{:08x}",hr as u32))}
    let factory=Com(raw);let mut result=Vec::new();
    unsafe{
        let next:unsafe extern "system" fn(*mut c_void,u32,*mut *mut c_void)->i32=std::mem::transmute(factory.slot(12));
        for ordinal in 0..32 {
            let mut raw=std::ptr::null_mut();let hr=next(factory.0,ordinal,&mut raw);
            if hr as u32==0x887a0002{break}
            if hr<0||raw.is_null(){return Err(format!("DXGI adapter inventory failed: 0x{:08x}",hr as u32))}
            let adapter=Com(raw);
            let describe:unsafe extern "system" fn(*mut c_void,*mut Desc)->i32=std::mem::transmute(adapter.slot(10));
            let mut desc:Desc=std::mem::zeroed();let hr=describe(adapter.0,&mut desc);
            if hr<0{continue}
            let end=desc.name.iter().position(|&c|c==0).unwrap_or(desc.name.len());
            result.push(Adapter{name:String::from_utf16_lossy(&desc.name[..end]),vendor_id:desc.vendor,device_id:desc.device,
                dedicated_bytes:desc.dedicated as u64,shared_bytes:desc.shared as u64,
                luid:format!("{:08x}:{:08x}",desc.luid_high as u32,desc.luid_low),software:desc.flags&2!=0});
        }
    }
    Ok(result)
}

#[cfg(not(windows))]
pub fn enumerate()->Result<Vec<Adapter>,String>{Err("Native device routing requires Windows".into())}

pub fn dedicated_for(adapters:&[Adapter],vendor:u32)->u64 {
    adapters.iter().filter(|a|a.vendor_id==vendor&&!a.software).map(|a|a.dedicated_bytes).max().unwrap_or(0)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn routing_inventory_never_treats_software_as_compute_gpu(){
        let a=|software,bytes|Adapter{name:"fixture".into(),vendor_id:0x1002,device_id:1,dedicated_bytes:bytes,shared_bytes:0,luid:"0:0".into(),software};
        assert_eq!(dedicated_for(&[a(true,999),a(false,100)],0x1002),100);
        assert_eq!(dedicated_for(&[a(false,100)],0x10de),0);
    }
}
