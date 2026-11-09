//! Private project-format limits and streaming deserialization helpers.
use serde::{de::{DeserializeSeed, Error, SeqAccess, Visitor}, Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, fmt, marker::PhantomData};

pub(crate) const MAX_NODES: usize = 250_000;
pub(crate) const MAX_PATCHES: usize = 2_000_000;
pub(crate) const MAX_RECORDS: usize = 2_000_000;
pub(crate) const MAX_CHECKPOINT_PRIMITIVES: usize = 1_000_000;
pub(crate) const MAX_CANVAS: usize = 100_000;
pub(crate) const MAX_TEXT: usize = 1 << 20;
pub(crate) const MAX_TITLE: usize = 1 << 16;
pub(crate) const MAX_COMPRESSED: u64 = 128 << 20;
pub(crate) const MAX_JSON: u64 = 512 << 20;
pub(crate) const MAX_WORK: usize = 50_000_000;
pub(crate) const NODES:u8=0;pub(crate) const PATCHES:u8=1;pub(crate) const RECORDS:u8=2;
pub(crate) const CHECKPOINT_PRIMITIVES:u8=3;pub(crate) const CURRENT_PRIMITIVES:u8=4;
pub(crate) const EDGES:u8=5;pub(crate) const TEXT:u8=6;pub(crate) const RAW_ENTRIES:u8=7;
pub(crate) const OVERLAYS:u8=8;pub(crate) const CHECKPOINTS:u8=9;
pub(crate) const MANUAL_REVIEWS:u8=10;
pub(crate) const MODEL_IDS:u8=11;

#[derive(Clone,Copy)]
pub(crate) struct Limits { pub counts:[usize;12],pub compressed:u64,pub json:u64,pub work:usize }
impl Default for Limits { fn default()->Self{Self{counts:[MAX_NODES,MAX_PATCHES,MAX_RECORDS,
    MAX_CHECKPOINT_PRIMITIVES,MAX_CANVAS,MAX_NODES,128<<20,MAX_NODES,MAX_NODES,MAX_NODES,MAX_NODES,MAX_NODES],
    compressed:MAX_COMPRESSED,json:MAX_JSON,work:MAX_WORK}} }
pub(crate) struct Budget { counts:[usize;12],limits:Limits }
impl Budget {
    pub fn new(limits:Limits)->Self{Self{counts:[0;12],limits}}
    pub fn charge(&mut self,kind:u8,amount:usize)->Result<(),String>{let k=kind as usize;
        let value=self.counts[k].checked_add(amount).ok_or("项目资源计数溢出")?;
        if value>self.limits.counts[k]{return Err(format!("项目资源超出安全上限（类别{kind}）"));}
        self.counts[k]=value;Ok(())
    }
}
thread_local!{static READ_BUDGET:RefCell<Option<Budget>>=const{RefCell::new(None)};}
pub(crate) struct ReadGuard;
impl ReadGuard {pub fn begin(limits:Limits)->Result<Self,String>{READ_BUDGET.with(|slot|{
    let mut slot=slot.borrow_mut();if slot.is_some(){return Err("不能嵌套读取项目文件".into());}
    *slot=Some(Budget::new(limits));Ok(Self)
})}}
impl Drop for ReadGuard{fn drop(&mut self){READ_BUDGET.with(|slot|*slot.borrow_mut()=None);}}
fn charge_read<E:Error>(kind:u8,amount:usize)->Result<(),E>{READ_BUDGET.with(|slot|{
    if let Some(budget)=slot.borrow_mut().as_mut(){budget.charge(kind,amount).map_err(E::custom)}else{Ok(())}
})}

pub(crate) struct Counted<T,const KIND:u8>(pub T);
impl<'de,T:Deserialize<'de>,const KIND:u8>Deserialize<'de>for Counted<T,KIND>{
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{charge_read::<D::Error>(KIND,1)?;Ok(Self(T::deserialize(d)?))}
}
pub(crate) struct LimitedVec<T,const MAX:usize>(pub Vec<T>);
impl<'de,T:Deserialize<'de>,const MAX:usize>Deserialize<'de>for LimitedVec<T,MAX>{
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct Items<T,const MAX:usize>(PhantomData<T>);
        struct Item<T>{full:bool,marker:PhantomData<T>}
        impl<'de,T:Deserialize<'de>>DeserializeSeed<'de>for Item<T>{type Value=T;
            fn deserialize<D:Deserializer<'de>>(self,d:D)->Result<T,D::Error>{
                if self.full{return Err(D::Error::custom("项目数组超过安全上限"));}T::deserialize(d)
            }}
        impl<'de,T:Deserialize<'de>,const MAX:usize>Visitor<'de>for Items<T,MAX>{type Value=LimitedVec<T,MAX>;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{write!(f,"bounded project array")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                if seq.size_hint().is_some_and(|n|n>MAX){return Err(A::Error::custom("项目数组超过安全上限"));}
                let mut result=Vec::new();
                while let Some(value)=seq.next_element_seed(Item{full:result.len()>=MAX,marker:PhantomData})?{
                    result.try_reserve(1).map_err(A::Error::custom)?;result.push(value);
                }Ok(LimitedVec(result))
            }}
        d.deserialize_seq(Items::<T,MAX>(PhantomData))
    }
}
fn text<'de,D:Deserializer<'de>>(d:D,max:usize)->Result<String,D::Error>{let s=String::deserialize(d)?;
    if s.len()>max{return Err(D::Error::custom("项目文本超过安全上限"));}charge_read::<D::Error>(TEXT,s.len())?;Ok(s)
}
pub(crate) fn read_text<'de,D:Deserializer<'de>>(d:D)->Result<String,D::Error>{text(d,MAX_TEXT)}
pub(crate) fn read_title<'de,D:Deserializer<'de>>(d:D)->Result<String,D::Error>{text(d,MAX_TITLE)}
pub(crate) fn strict_settings<'de,D:Deserializer<'de>>(d:D)->Result<crate::document::Settings,D::Error>{
    #[derive(Deserialize)]#[serde(deny_unknown_fields)]struct Accel{
        mode:crate::acceleration::AccelerationMode,fusion:crate::acceleration::FeaturePreference,
        split_k:crate::acceleration::FeaturePreference,primitive_graph:crate::acceleration::FeaturePreference,
        kv_int8:crate::acceleration::FeaturePreference,
    }
    #[derive(Deserialize)]#[serde(deny_unknown_fields)]struct Settings{
        cfg:f32,renyi_alpha:f32,seed:u64,parallel_sequences:usize,
        #[serde(default="super::default_auto_seed")]auto_seed:bool,
        sampling_backend:crate::document::SamplingBackend,context_overclock:bool,pause_at_stage_end:bool,
        #[serde(default)]half_region_switch:bool,
        auto_reorder_before_generation:bool,acceleration:Accel,
    }
    let value=Settings::deserialize(d)?;
    let settings=crate::document::Settings{cfg:value.cfg,renyi_alpha:value.renyi_alpha,seed:value.seed,
        auto_seed:value.auto_seed,
        parallel_sequences:value.parallel_sequences,sampling_backend:value.sampling_backend,
        context_overclock:value.context_overclock,pause_at_stage_end:value.pause_at_stage_end,
        half_region_switch:value.half_region_switch,
        auto_reorder_before_generation:value.auto_reorder_before_generation,
        acceleration:crate::acceleration::AccelerationSettings{mode:value.acceleration.mode,fusion:value.acceleration.fusion,
            split_k:value.acceleration.split_k,primitive_graph:value.acceleration.primitive_graph,kv_int8:value.acceleration.kv_int8}};
    settings.validate().map_err(D::Error::custom)?;Ok(settings)
}

#[derive(Clone,Default)]
pub(crate) struct IdProjection { hash:Sha256,pub count:usize }
impl IdProjection{
    pub fn push(&mut self,id:u64)->Result<(),String>{if id==0{return Err("模型ID映射包含零对象编号".into());}
        self.hash.update(id.to_le_bytes());self.count=self.count.checked_add(1).ok_or("项目对象编号计数溢出")?;Ok(())}
    pub fn same(&self,other:&Self)->bool{self.count==other.count&&self.hash.clone().finalize()==other.hash.clone().finalize()}
}
#[derive(Clone)]
pub(crate) struct Projection { hash:Sha256,pub count:usize,ids:IdProjection }
impl Default for Projection{fn default()->Self{Self{hash:Sha256::new(),count:0,ids:IdProjection::default()}}}
impl Projection{
    pub fn push(&mut self,row:&[u32;9])->Result<(),String>{for value in row{self.hash.update(value.to_le_bytes());}
        self.count=self.count.checked_add(1).ok_or("项目图元计数溢出")?;Ok(())}
    pub fn same(&self,other:&Self)->bool{self.count==other.count&&self.hash.clone().finalize()==other.hash.clone().finalize()}
    pub fn push_with_id(&mut self,row:&[u32;9],id:u64)->Result<(),String>{self.push(row)?;self.ids.push(id)}
    pub fn ordered_ids(&self)->&IdProjection{&self.ids}
}
pub(crate) struct Work { remaining:usize,limit:usize }
impl Work{pub fn new(limit:usize)->Self{Self{remaining:limit,limit}}
    pub fn charge(&mut self,n:usize)->Result<(),String>{self.remaining=self.remaining.checked_sub(n)
        .ok_or_else(||format!("存档完整性校验达到计算量上限（已用{}，本次需要{n}，上限{}）；这不是标准模式切换检查",
            self.limit-self.remaining,self.limit))?;Ok(())}}
