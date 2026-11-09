//! Small single-image adapter around the original training WGSL search engine.
use eframe::wgpu::{self,util::DeviceExt};
use std::sync::{Arc,Mutex,mpsc};
use bytemuck::{Pod,Zeroable};
#[path="types.rs"]mod original_types;
pub(super) use original_types::{GpuShape,ImageState};
use original_types::Config;

pub(super) struct Engine{
    pub device:wgpu::Device,pub queue:wgpu::Queue,pipeline:wgpu::ComputePipeline,bind:wgpu::BindGroup,
    pub canvas:wgpu::Buffer,pub states:wgpu::Buffer,pub history:wgpu::Buffer,pub name:String,
    error:Arc<Mutex<Option<String>>>,
}
impl Engine{
    pub async fn new(shader:&str,target:&[u32],bg:u32,state:&ImageState)->Result<Self,String>{
        let instance=wgpu::Instance::new(wgpu::InstanceDescriptor{backends:wgpu::Backends::DX12|wgpu::Backends::VULKAN,..Default::default()});
        let mut adapters=instance.enumerate_adapters(wgpu::Backends::DX12|wgpu::Backends::VULKAN);
        adapters.sort_by_key(|a|{let info=a.get_info();(match info.device_type{wgpu::DeviceType::DiscreteGpu=>0,wgpu::DeviceType::IntegratedGpu=>1,wgpu::DeviceType::VirtualGpu=>2,_=>3},if info.backend==wgpu::Backend::Dx12{0}else{1})});
        let mut errors=Vec::new();
        for adapter in adapters{
            if adapter.get_info().device_type==wgpu::DeviceType::Cpu{continue;}
            match Self::from_adapter(adapter,shader,target,bg,state).await{Ok(engine)=>return Ok(engine),Err(error)=>errors.push(error)}
        }
        Err(format!("没有可用于参考序列化的DX12/Vulkan显卡。{}",errors.join("；")))
    }
    async fn from_adapter(adapter:wgpu::Adapter,shader:&str,target:&[u32],bg:u32,state:&ImageState)->Result<Self,String>{
        let info=adapter.get_info();let mut limits=wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        limits.max_storage_buffers_per_shader_stage=6;
        let(device,queue)=adapter.request_device(&wgpu::DeviceDescriptor{label:Some("Reference serializer"),required_features:wgpu::Features::empty(),required_limits:limits},None)
            .await.map_err(|e|format!("{}设备初始化失败：{e}",info.name))?;
        let error=Arc::new(Mutex::new(None));let callback=error.clone();
        device.on_uncaptured_error(Box::new(move|e:wgpu::Error|{if let Ok(mut slot)=callback.lock(){*slot=Some(e.to_string());}}));
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("Original primitive search"),source:wgpu::ShaderSource::Wgsl(shader.into())});
        let layout=device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:None,entries:&std::array::from_fn::<_,6,_>(|index|wgpu::BindGroupLayoutEntry{
            binding:index as u32,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{
                ty:if index==0{wgpu::BufferBindingType::Uniform}else{wgpu::BufferBindingType::Storage{read_only:index==1||index==5}},
                has_dynamic_offset:false,min_binding_size:None},count:None})});
        let pipeline_layout=device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:None,bind_group_layouts:&[&layout],push_constant_ranges:&[]});
        let pipeline=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:Some(&pipeline_layout),module:&module,entry_point:"main"});
        if let Some(e)=device.pop_error_scope().await{return Err(format!("{}序列化着色器初始化失败：{e}",info.name));}
        let config=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("Reference single image config"),contents:bytemuck::bytes_of(&Config{image_offset:0,dispatch_images:1,num_images:1,pad:0}),usage:wgpu::BufferUsages::UNIFORM});
        let target=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("Reference target"),contents:bytemuck::cast_slice(target),usage:wgpu::BufferUsages::STORAGE});
        let canvas=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("Reference canvas"),contents:bytemuck::cast_slice(&vec![bg;512*512]),usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST});
        let states=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("Reference state"),contents:bytemuck::bytes_of(state),usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST});
        let history=device.create_buffer(&wgpu::BufferDescriptor{label:Some("Reference history"),size:895*std::mem::size_of::<GpuShape>()as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let background=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("Reference background"),contents:bytemuck::bytes_of(&bg),usage:wgpu::BufferUsages::STORAGE});
        let buffers=[&config,&target,&canvas,&states,&history,&background];
        let bind=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&layout,entries:&std::array::from_fn::<_,6,_>(|i|wgpu::BindGroupEntry{binding:i as u32,resource:buffers[i].as_entire_binding()})});
        Ok(Self{device,queue,pipeline,bind,canvas,states,history,name:info.name,error})
    }
    pub fn step(&self)->Result<ImageState,String>{
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(1,1,1);}
        self.queue.submit(Some(encoder.finish()));self.read(&self.states,0)
    }
    pub fn read<T:Pod>(&self,source:&wgpu::Buffer,offset:u64)->Result<T,String>{
        if let Some(error)=self.error.lock().ok().and_then(|s|s.clone()){return Err(format!("参考序列化GPU错误：{error}"));}
        let size=std::mem::size_of::<T>()as u64;
        let staging=self.device.create_buffer(&wgpu::BufferDescriptor{label:Some("Reference readback"),size,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());encoder.copy_buffer_to_buffer(source,offset,&staging,0,size);self.queue.submit(Some(encoder.finish()));
        let slice=staging.slice(..);let(send,recv)=mpsc::sync_channel(1);slice.map_async(wgpu::MapMode::Read,move|result|{let _=send.send(result);});
        self.device.poll(wgpu::Maintain::Wait);
        if let Some(error)=self.error.lock().ok().and_then(|s|s.clone()){return Err(format!("参考序列化GPU错误：{error}"));}
        recv.recv().map_err(|e|e.to_string())?.map_err(|e|format!("参考序列化GPU回读失败：{e}"))?;
        let bytes=slice.get_mapped_range();let value=bytemuck::pod_read_unaligned(&bytes);drop(bytes);staging.unmap();Ok(value)
    }
    pub fn update(&self,state:&ImageState,pixels:Option<&[u32]>){
        if let Some(pixels)=pixels{self.queue.write_buffer(&self.canvas,0,bytemuck::cast_slice(pixels));}
        self.queue.write_buffer(&self.states,0,bytemuck::bytes_of(state));
    }
}

pub(super) fn empty_state(sse:f64)->ImageState{let mut state=ImageState::zeroed();state.initial_sse=sse as f32;state.current_sse=sse as f32;state.phase_two_start_count=85;state.rng=0x52ef_a937;state}
