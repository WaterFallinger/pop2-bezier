//! Native tensor implementation of the project's compressed-primitive GPT.
//!
//! No Python interpreter, Python service, TorchScript export, or substitute LLM
//! is involved. The original checkpoint is read by Candle's restricted pickle
//! tensor reader, then evaluated by this Rust implementation.

use anyhow::{bail, Context, Result};
use candle_core::{pickle::PthTensors, DType, Device, Tensor};
use std::{path::Path, sync::Mutex};

mod batch;
mod batch_geometry;
mod cache;
pub(crate) mod cache_probe;
mod cache_policy;
mod decode_metadata;
mod device_prediction;
mod diagnostics;
pub(crate) mod graph_decode;
mod write_plan;
mod fused_attention;
#[cfg(feature = "rocm")]
pub(crate) mod hip_kernel;
#[cfg(feature = "rocm")]
pub(crate) mod hip_attention;
mod graph_ops;
mod int8_cache;
mod int8_kv_ops;
mod int8_policy;
mod truncate;
pub use int8_policy::Int8KvStatus;
#[cfg(test)]
pub(crate) use batch::tests::small_model as small_test_model;
#[path = "../cuda_compat.rs"]
pub(crate) mod cuda_compat;
use cache::LayerCache;
pub use cache_policy::CachePlan;
pub(crate) use batch_geometry::checked_batch_rows;
use decode_metadata::DecodeMetadataCache;
pub use device_prediction::{finite_rows, DevicePrediction};
pub use diagnostics::FixedDecodeCase;

pub const OFFSETS: [u32; 9] = [0, 1024, 2048, 2318, 4366, 6414, 6670, 6798, 6926];
pub const BINS: [u32; 9] = [1024, 1024, 270, 2048, 2048, 256, 128, 128, 128];
pub const ATTRIBUTE_VOCAB: usize = 7054;
pub const PAD: u32 = 7054;
pub const REGION_MARKER: u32 = 7057;
pub const TRUE_END: u32 = 7066;
pub const INTERRUPT: u32 = 7067;
pub const BOS: u32 = 7068;
pub const PARTIAL: [u32; 9] = [PAD, 7058, 7059, 7060, 7061, 7062, 7063, 7064, 7065];
pub const BOS_RECORD: [u32; 9] = [BOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const END_RECORD: [u32; 9] = [TRUE_END, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const INTERRUPT_RECORD: [u32; 9] = [INTERRUPT, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const MAX_CONTEXT: usize = 1024;

struct Block {
    norm1: Tensor,
    qkv: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    projection: Tensor,
    norm2: Tensor,
    gate_up: Tensor,
    down: Tensor,
}

#[derive(Clone)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    pub length: usize,
    position_offset: usize,
    storage_dtype: DType,
}

impl KvCache {
    pub fn int8_kv_enabled(&self) -> bool { self.storage_dtype == DType::U8 }
    pub fn reserved_kv_bytes(&self) -> usize { self.layers.iter().map(|layer| layer.reserved_bytes()).sum() }
    pub fn reserved_int8_scale_bytes(&self) -> usize { self.layers.iter().map(|layer| layer.int8_scale_bytes()).sum() }
    pub fn layer_lengths(&self) -> Result<Vec<usize>> {
        self.layers.iter().map(|layer| Ok(layer.len())).collect()
    }
    pub fn absolute_position(&self) -> usize {
        self.length + self.position_offset
    }
}

#[derive(Clone)]
pub struct Prediction {
    pub attributes: Vec<f32>,
    pub termination: [f32; 2],
    pub attribute_offset: usize,
}

impl Prediction {
    pub fn attribute(&self, id: u32) -> Result<f64> {
        let index = (id as usize)
            .checked_sub(self.attribute_offset)
            .context("requested attribute precedes projected head")?;
        self.attributes
            .get(index)
            .copied()
            .map(f64::from)
            .context("requested attribute is outside projected head")
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OutputScope {
    All,
    NextPrimitive,
    Field(usize),
}

impl OutputScope {
    fn range(self) -> Result<(usize, usize, bool)> {
        Ok(match self {
            Self::All => (0, ATTRIBUTE_VOCAB, true),
            Self::NextPrimitive => (0, BINS[0] as usize, true),
            Self::Field(field) if field < 9 => {
                (OFFSETS[field] as usize, BINS[field] as usize, false)
            }
            Self::Field(_) => bail!("attribute head index must be 0..8"),
        })
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Architecture {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_width: usize,
    pub ffn_hidden: usize,
    pub device: String,
    pub dtype: String,
}

pub struct PrimitiveModel {
    pub architecture: Architecture,
    pub device: Device,
    dtype: DType,
    embedding: Tensor,
    text_embedding: Tensor,
    blocks: Vec<Block>,
    final_norm: Tensor,
    termination_weight: Tensor,
    termination_bias: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    diagnostic_copy_transposed_keys: bool,
    diagnostic_stack_kv: bool,
    diagnostic_legacy_metadata: bool,
    fp32_kv_storage: bool,
    int8_kv: Int8KvStatus,
    optional_fusion: bool,
    optional_split_k: bool,
    #[cfg(feature="cuda")]
    cuda_auto_attention_rows: Option<usize>,
    #[cfg(feature = "rocm")]
    hip_auto_attention_rows: Option<usize>,
    decode_metadata: Mutex<DecodeMetadataCache>,
}

impl PrimitiveModel {
    #[cfg(test)]
    pub(crate) fn force_test_termination(&mut self, end: bool) -> Result<()> {
        self.termination_weight = Tensor::zeros((2, self.architecture.width), self.dtype, &self.device)?;
        self.termination_bias = Tensor::from_vec(if end { vec![-20f32,20.] } else { vec![20f32,-20.] }, 2, &self.device)?.to_dtype(self.dtype)?;
        Ok(())
    }
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "rocm")]
    fn strict_math_guard(&self) -> Option<candle_core::rocm_backend::strict::DeviceOnlyGuard> {
        self.device.is_rocm().then(candle_core::rocm_backend::strict::DeviceOnlyGuard::enter)
    }

    #[allow(unexpected_cfgs)]
    #[cfg(not(feature = "rocm"))]
    fn strict_math_guard(&self) {}

    pub fn load(path: &Path, use_cuda: bool, progress: impl FnMut(String)) -> Result<Self> {
        if !path.is_file() {
            bail!("模型文件不存在：{}", path.display());
        }
        let device = if use_cuda {
            #[cfg(feature = "cuda")]
            {
                let ordinal=cuda_compat::selected_cuda_ordinal().map_err(anyhow::Error::msg)?;
                Device::new_cuda(ordinal)
                    .context("无法初始化 NVIDIA CUDA；请检查驱动和原生 CUDA 运行库")?
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("此可执行文件未启用 CUDA，请使用带 CUDA 的构建；不会静默退回 CPU");
            }
        } else {
            Device::Cpu
        };
        let dtype = if device.is_cuda() {
            #[cfg(feature="cuda")]
            {
                let actual=device.as_cuda_device()?.cuda_stream().context().compute_capability()?;
                if !cuda_compat::CudaKernelProfile::PortableSm80.accepts(actual.0,actual.1) {
                    bail!("所选CUDA设备的计算能力为{}.{}，本GPU模型路径要求原生BF16（8.0及更新）；请选择CPU兼容后端",actual.0,actual.1);
                }
            }
            DType::BF16
        } else {
            DType::F32
        };
        Self::load_on_device(path, device, dtype, progress)
    }

    /// Explicit device entry point for isolated providers. The normal loader
    /// retains its existing CPU/CUDA selection and precision policy.
    pub fn load_on_device(
        path: &Path,
        device: Device,
        dtype: DType,
        mut progress: impl FnMut(String),
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("模型文件不存在：{}", path.display());
        }
        #[cfg(feature="cuda")]
        if device.is_cuda() {
            let actual=device.as_cuda_device()?.cuda_stream().context().compute_capability()?;
            let profile=cuda_compat::compiled_kernel_profile();
            if !profile.accepts(actual.0,actual.1) {
                bail!("所选CUDA设备{}.{}与此GPU后端的{}内核不匹配；请选择合适的通用GPU或CPU兼容后端",actual.0,actual.1,profile.embedded_target());
            }
            if dtype==DType::BF16 && actual.0<8 {
                bail!("此GPU不支持本模型所需的原生BF16；请选择CPU兼容后端");
            }
            // Admission must exercise the selected runtime's ordinary GEMM,
            // not just trust its architecture string. This happens before
            // reading/allocating the large model and never changes devices.
            let values=(0..256).map(|i|(i%16+1) as f32).collect::<Vec<_>>();
            let identity=(0..256).map(|i|if i/16==i%16{1f32}else{0f32}).collect::<Vec<_>>();
            let a=Tensor::from_vec(values.clone(),(16,16),&device)?.to_dtype(dtype)?;
            let b=Tensor::from_vec(identity,(16,16),&device)?.to_dtype(dtype)?;
            let checked=a.matmul(&b)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
                .context("所选CUDA设备的普通矩阵计算验证失败")?;
            if checked != values { bail!("所选CUDA设备普通矩阵计算结果未通过验证，停止GPU加载"); }
        }
        progress(format!("读取原始权重：{}", path.display()));
        let loader = match PthTensors::new(path, Some("ema_state_dict")) {
            Ok(loader) if !loader.tensor_infos().is_empty() => loader,
            _ => PthTensors::new(path, Some("model_state_dict"))
                .context("无法读取 checkpoint 的 ema_state_dict/model_state_dict")?,
        };
        let infos = loader.tensor_infos();
        let prefix = if infos.contains_key("wte.weight") {
            ""
        } else if infos.contains_key("module.wte.weight") {
            "module."
        } else if infos.contains_key("_orig_mod.wte.weight") {
            "_orig_mod."
        } else {
            bail!("checkpoint 缺少本项目的 wte.weight；不支持用其他模型代替");
        };
        let read = |name: &str| -> Result<Tensor> {
            loader
                .get(&format!("{prefix}{name}"))?
                .with_context(|| format!("权重缺失：{name}"))?
                .to_device(&device)?
                .to_dtype(dtype)
                .map_err(Into::into)
        };
        let embedding = read("wte.weight")?;
        let text_embedding = read("text_wte.weight")?;
        let (vocab, width) = embedding.dims2()?;
        if vocab != 7069 || text_embedding.dims2()?.0 != 1500 {
            bail!(
                "不兼容的词表：图元 {vocab}，文本 {}（需要 7069/1500）",
                text_embedding.dims2()?.0
            );
        }
        let mut layers = 0;
        while infos.contains_key(&format!("{prefix}h.{layers}.attn.c_attn.weight")) {
            layers += 1;
        }
        if layers == 0 {
            bail!("checkpoint 没有 Transformer 层");
        }
        let head_width = read("h.0.attn.q_norm.weight")?.elem_count();
        if width % head_width != 0 || head_width % 2 != 0 {
            bail!("不兼容的注意力头维度");
        }
        let heads = width / head_width;
        let ffn_hidden = infos
            .get(&format!("{prefix}h.0.mlp.gate_up.weight"))
            .context("缺少 SwiGLU 权重")?
            .layout
            .dims()[0]
            / 2;
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            progress(format!("加载 Transformer 层 {}/{}", i + 1, layers));
            let key = |suffix: &str| format!("h.{i}.{suffix}");
            blocks.push(Block {
                norm1: read(&key("norm_1.weight"))?,
                qkv: read(&key("attn.c_attn.weight"))?,
                q_norm: read(&key("attn.q_norm.weight"))?,
                k_norm: read(&key("attn.k_norm.weight"))?,
                projection: read(&key("attn.c_proj.weight"))?,
                norm2: read(&key("norm_2.weight"))?,
                gate_up: read(&key("mlp.gate_up.weight"))?,
                down: read(&key("mlp.down.weight"))?,
            });
        }
        let final_norm = read("final_norm.weight")?;
        let termination_weight = read("termination_head.weight")?;
        let termination_bias = read("termination_head.bias")?;
        let mut cos = Vec::with_capacity(MAX_CONTEXT * head_width / 2);
        let mut sin = Vec::with_capacity(cos.capacity());
        for pos in 0..MAX_CONTEXT {
            for pair in 0..head_width / 2 {
                let inverse_frequency =
                    1.0f32 / 10000f32.powf((pair * 2) as f32 / head_width as f32);
                let angle = pos as f32 * inverse_frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let rope_cos =
            Tensor::from_vec(cos, (MAX_CONTEXT, head_width / 2), &device)?.to_dtype(dtype)?;
        let rope_sin =
            Tensor::from_vec(sin, (MAX_CONTEXT, head_width / 2), &device)?.to_dtype(dtype)?;
        let architecture = Architecture {
            width,
            layers,
            heads,
            head_width,
            ffn_hidden,
            device: if device.is_cuda() {
                #[cfg(feature="cuda")]
                { format!("CUDA:{}",device.as_cuda_device()?.cuda_stream().context().ordinal()) }
                #[cfg(not(feature="cuda"))]
                { "CUDA（未启用本后端）".into() }
            } else if device.is_cpu() {
                "CPU".into()
            } else {
                format!("{:?}", device.location())
            },
            dtype: format!("{dtype:?}"),
        };
        progress(format!(
            "原生模型已加载：{layers} 层 / {width} 宽 / {heads} 头 / {dtype:?}"
        ));
        Ok(Self {
            architecture,
            device,
            dtype,
            embedding,
            text_embedding,
            blocks,
            final_norm,
            termination_weight,
            termination_bias,
            rope_cos,
            rope_sin,
            diagnostic_copy_transposed_keys: false,
            diagnostic_stack_kv: false,
            diagnostic_legacy_metadata: false,
            fp32_kv_storage: false,
            int8_kv: Int8KvStatus::default(),
            optional_fusion: false,
            optional_split_k: false,
            #[cfg(feature="cuda")]
            cuda_auto_attention_rows:None,
            #[cfg(feature = "rocm")]
            hip_auto_attention_rows: None,
            decode_metadata: Mutex::new(DecodeMetadataCache::default()),
        })
    }

    pub fn new_cache(&self) -> KvCache {
        self.new_cache_offset(0)
    }

    pub fn new_cache_offset(&self, position_offset: usize) -> KvCache {
        KvCache {
            layers: vec![LayerCache::default(); self.blocks.len()],
            length: 0,
            position_offset,
            storage_dtype: if self.int8_kv_enabled() {
                DType::U8
            } else if self.fp32_kv_storage {
                DType::F32
            } else {
                self.dtype
            },
        }
    }

    pub fn configure_cache_for_request(
        &mut self,
        lanes: usize,
        cfg: bool,
        prefix_records: usize,
        max_new: usize,
    ) -> Result<CachePlan> {
        // Drain retired request buffers before observing memory. No lane or CFG
        // setting is changed by this decision; unknown budgets keep compact KV.
        self.device.synchronize()?;
        #[cfg(feature = "cuda")]
        let available = self
            .device
            .as_cuda_device()
            .ok()
            .and_then(|d| d.cuda_stream().context().mem_get_info().ok())
            .map(|(free, _)| free as u64);
        #[cfg(not(feature = "cuda"))]
        let available = None;
        let mut plan = cache_policy::plan_cache_for_workload(
            cache_policy::CacheBudgetInput {
                available_bytes: available,
                layers: self.architecture.layers,
                width: self.architecture.width,
                max_context: MAX_CONTEXT,
                logical_lanes: lanes,
                cfg_enabled: cfg,
                headroom_bytes: Some(2u64 << 30),
            },
            cache_policy::planned_context_records(prefix_records, max_new, MAX_CONTEXT),
        );
        if self.int8_kv_enabled() {
            plan.use_fp32 = false;
            plan.reason = cache_policy::CachePlanReason::Int8History;
        }
        self.fp32_kv_storage = plan.use_fp32;
        // The smallest physical batches did not show a stable metadata win in
        // the controlled trials. Keep their established path for now.
        self.diagnostic_legacy_metadata = plan.physical_rows.is_none_or(|n| n < 4);
        Ok(plan)
    }

    pub fn prepare_optional_kernels(&mut self, fusion: bool, split_k: bool) -> Result<()> {
        self.optional_fusion = false;
        self.optional_split_k = false;
        if !(fusion || split_k) { return Ok(()); }
        if !self.supports_native_decode_kernels() || self.architecture.head_width != 64 {
            bail!("当前模型后端不支持此融合解码路径");
        }
        let q = Tensor::zeros((1,1,64),DType::F32,&self.device)?;
        let k = Tensor::zeros((1,257,64),self.dtype,&self.device)?;
        let v = Tensor::zeros((1,257,64),self.dtype,&self.device)?;
        let out = fused_attention::decode_attention(&q,&k,&v,None,1,if split_k {128}else{0})?;
        let checked = out.flatten_all()?.to_vec1::<f32>()?;
        if checked.iter().any(|v| !v.is_finite() || v.abs()>1e-6) { bail!("融合注意力准备验证失败"); }
        if fusion && self.device.is_cuda() {
            let gate = Tensor::zeros((1,self.architecture.ffn_hidden*2),self.dtype,&self.device)?;
            let checked = graph_ops::swiglu(&gate)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            if checked.iter().any(|v| !v.is_finite() || v.abs()>1e-6) { bail!("融合FFN准备验证失败"); }
        }
        self.optional_fusion = fusion;
        self.optional_split_k = split_k;
        Ok(())
    }

    pub(crate) fn supports_native_decode_kernels(&self) -> bool {
        #[cfg(feature = "rocm")]
        if self.device.is_rocm() { return true; }
        cfg!(feature = "cuda") && self.device.is_cuda()
    }

    pub(crate) fn optional_decode_features(&self) -> (bool, bool) {
        (self.optional_fusion, self.optional_split_k)
    }
    pub(crate) fn current_fp32_kv(&self) -> bool { self.fp32_kv_storage }
    #[cfg(feature="cuda")]
    pub(crate) fn restrict_cuda_auto_attention(&mut self,physical_rows:Option<usize>) {
        self.cuda_auto_attention_rows=physical_rows;
    }

    #[cfg(feature = "rocm")]
    pub(crate) fn restrict_hip_auto_attention(&mut self, physical_rows: Option<usize>) {
        self.hip_auto_attention_rows = physical_rows;
    }

    pub(crate) fn optional_attention_for_rows(&self, physical_rows: usize) -> bool {
        #[cfg(feature="cuda")]
        if self.device.is_cuda() && self.cuda_auto_attention_rows.is_some_and(|rows|rows!=physical_rows){return false;}
        #[cfg(feature = "rocm")]
        if self.device.is_rocm() && self.hip_auto_attention_rows.is_some_and(|rows| rows != physical_rows) {
            return false;
        }
        let _ = physical_rows;
        self.supports_native_decode_kernels() && (self.optional_fusion || self.optional_split_k)
    }

    fn primitive_embedding(&self, record: &[u32; 9]) -> Result<Tensor> {
        self.primitive_embeddings(&[*record])
    }

    fn primitive_embeddings_device(&self, records: &Tensor) -> Result<Tensor> {
        let _device_only = self.strict_math_guard();
        let (batch, fields) = records.dims2()?;
        if batch == 0
            || fields != 9
            || records.dtype() != DType::U32
            || !records.device().same_device(&self.device)
        {
            bail!("device primitive records require nonempty U32[B,9] on the model device");
        }
        // Generation records contain ordinary/partial primitives, never BOS,
        // region markers or termination controls; those keep the prefill path.
        Ok(self
            .embedding
            .index_select(&records.flatten_all()?, 0)?
            .reshape((batch, 9, self.architecture.width))?
            .sum(1)?)
    }

    fn primitive_embeddings(&self, records: &[[u32; 9]]) -> Result<Tensor> {
        let _device_only = self.strict_math_guard();
        let indices: Vec<_> = records.iter().flatten().copied().collect();
        let indices = Tensor::from_vec(indices, records.len() * 9, &self.device)?;
        // The trained representation is the literal sum of all nine lookups,
        // including the eight distinct, learned empty-attribute embeddings.
        let embedded = self
            .embedding
            .index_select(&indices, 0)?
            .reshape((records.len(), 9, self.architecture.width))?
            .sum(1)?;
        let controls: Vec<u8> = records
            .iter()
            .map(|r| u8::from([BOS, TRUE_END, INTERRUPT].contains(&r[0])))
            .collect();
        if controls.iter().all(|&v| v == 0) {
            return Ok(embedded);
        }
        let first: Vec<u32> = records.iter().map(|r| r[0]).collect();
        let first = Tensor::from_vec(first, records.len(), &self.device)?;
        let control_embedding = self.embedding.index_select(&first, 0)?;
        let mask = Tensor::from_vec(controls, (records.len(), 1), &self.device)?
            .broadcast_as((records.len(), self.architecture.width))?;
        Ok(mask.where_cond(&control_embedding, &embedded)?)
    }

    pub fn forward_record(
        &self,
        record: &[u32; 9],
        cache: &mut KvCache,
        commit: bool,
    ) -> Result<Prediction> {
        self.forward_record_for(record, cache, commit, OutputScope::All)
    }

    pub fn forward_record_for(
        &self,
        record: &[u32; 9],
        cache: &mut KvCache,
        commit: bool,
        scope: OutputScope,
    ) -> Result<Prediction> {
        let hidden = self.forward_embedding(self.primitive_embedding(record)?, cache, commit)?;
        self.predict_many(&hidden, scope)?
            .pop()
            .context("missing single prediction")
    }

    pub fn forward_text(&self, id: u32, cache: &mut KvCache) -> Result<()> {
        if id >= 1500 {
            bail!("文本 token 超出 1500 词表：{id}");
        }
        let embedding = self.text_embedding.narrow(0, id as usize, 1)?;
        let _ = self.forward_embedding(embedding, cache, true)?;
        Ok(())
    }

    pub fn prefill_text(&self, ids: &[u32], cache: &mut KvCache) -> Result<()> {
        let _device_only = self.strict_math_guard();
        if ids.is_empty() || ids.iter().any(|&id| id >= 1500) {
            bail!("文本 token 非法");
        }
        let indices = Tensor::from_slice(ids, ids.len(), &self.device)?;
        let embedding = self.text_embedding.index_select(&indices, 0)?;
        let _ = self.forward_embedding(embedding, cache, true)?;
        Ok(())
    }

    pub fn prefill_records(&self, records: &[[u32; 9]], cache: &mut KvCache) -> Result<Prediction> {
        self.prefill_records_for(records, cache, Some(OutputScope::All))?
            .context("missing prefill prediction")
    }

    pub fn prefill_records_for(
        &self,
        records: &[[u32; 9]],
        cache: &mut KvCache,
        scope: Option<OutputScope>,
    ) -> Result<Option<Prediction>> {
        if records.is_empty() {
            bail!("图元预填充不能为空");
        }
        let hidden = self.forward_embedding(self.primitive_embeddings(records)?, cache, true)?;
        match scope {
            Some(scope) => Ok(self
                .predict_many(&hidden.narrow(0, records.len() - 1, 1)?, scope)?
                .pop()),
            None => Ok(None),
        }
    }

    pub fn prefill_records_device(
        &self,
        records: &[[u32; 9]],
        cache: &mut KvCache,
        scope: OutputScope,
    ) -> Result<DevicePrediction> {
        if records.is_empty() {
            bail!("empty device prefill");
        }
        let hidden = self.forward_embedding(self.primitive_embeddings(records)?, cache, true)?;
        self.predict_device(&hidden.narrow(0, records.len() - 1, 1)?, scope)
    }

    fn forward_embedding(
        &self,
        mut x: Tensor,
        cache: &mut KvCache,
        commit: bool,
    ) -> Result<Tensor> {
        let _device_only = self.strict_math_guard();
        let position = cache.length + cache.position_offset;
        let query_length = x.dims2()?.0;
        if position + query_length > MAX_CONTEXT {
            bail!("当前上下文达到 {MAX_CONTEXT}，需要重建区域上下文");
        }
        let heads = self.architecture.heads;
        let head_width = self.architecture.head_width;
        let width = self.architecture.width;
        let int8_decode = cache.int8_kv_enabled() && query_length == 1;
        let int8_past = if int8_decode {
            Some(Tensor::from_vec(vec![u32::try_from(cache.length).context("INT8 history length exceeds U32")?], 1, &self.device)?)
        } else { None };
        let cos = self
            .rope_cos
            .narrow(0, position, query_length)?
            .contiguous()?;
        let sin = self
            .rope_sin
            .narrow(0, position, query_length)?
            .contiguous()?;
        let causal_mask = if query_length > 1 {
            let past_length = cache.length;
            let key_length = past_length + query_length;
            let mask: Vec<f32> = (0..query_length)
                .flat_map(|q| {
                    (0..key_length).map(move |k| {
                        if k <= past_length + q {
                            0.
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(
                mask,
                (1, query_length, key_length),
                &self.device,
            )?)
        } else {
            None
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm1, 1e-6)?;
            let qkv = normalized.matmul(&block.qkv.t()?)?;
            let q = qkv
                .narrow(1, 0, width)?
                .reshape((1, query_length, heads, head_width))?
                .transpose(1, 2)?;
            let k = qkv
                .narrow(1, width, width)?
                .reshape((1, query_length, heads, head_width))?
                .transpose(1, 2)?;
            let v = qkv
                .narrow(1, width * 2, width)?
                .reshape((query_length, heads, head_width))?
                .transpose(0, 1)?
                .contiguous()?;
            let q = candle_nn::ops::rms_norm(&q.contiguous()?, &block.q_norm, 1e-6)?;
            let k = candle_nn::ops::rms_norm(&k.contiguous()?, &block.k_norm, 1e-6)?;
            let q = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?.reshape((
                heads,
                query_length,
                head_width,
            ))?;
            let k = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?.reshape((
                heads,
                query_length,
                head_width,
            ))?;
            let layer = &mut cache.layers[i];
            let attended = if int8_decode {
                let history = layer.query_int8(&k, &v, cache.length, commit)?;
                int8_kv_ops::attention(&q.to_dtype(DType::F32)?.contiguous()?,
                    &history.key, &history.value, &history.key_scale, &history.value_scale,
                    &k.to_dtype(DType::F32)?.contiguous()?, &v.to_dtype(DType::F32)?.contiguous()?,
                    int8_past.as_ref().unwrap(), heads,
                    if self.optional_split_k && cache.length >= 255 { 128 } else { 0 })?
            } else {
                let (keys, values) = if cache.storage_dtype == DType::U8 {
                    layer.query_int8_prefill(&k, &v, cache.length, commit)?
                } else {
                    layer.query(&k.to_dtype(cache.storage_dtype)?, &v.to_dtype(cache.storage_dtype)?, cache.length, commit)?
                };
                if self.optional_attention_for_rows(1) && query_length==1 && head_width==64 {
                    fused_attention::decode_attention(&q.to_dtype(DType::F32)?, &keys, &values, None, heads,
                        if self.optional_split_k && keys.dim(1)? >= 256 {128} else {0})?
                } else {
                    let transposed_keys = keys.to_dtype(DType::F32)?.transpose(1, 2)?;
                    let transposed_keys = if self.diagnostic_copy_transposed_keys { transposed_keys.contiguous()? } else { transposed_keys };
                    let mut score = (q.to_dtype(DType::F32)?.matmul(&transposed_keys)? / (head_width as f64).sqrt())?;
                    if let Some(mask) = &causal_mask { score = score.broadcast_add(mask)?; }
                    candle_nn::ops::softmax_last_dim(&score)?.matmul(&values.to_dtype(DType::F32)?)?
                }
            };
            let attended = attended
                .to_dtype(self.dtype)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((query_length, width))?;
            x = (x + attended.matmul(&block.projection.t()?)?)?;
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm2, 1e-6)?;
            let gate_up = normalized.matmul(&block.gate_up.t()?)?;
            let gate = gate_up.narrow(1, 0, self.architecture.ffn_hidden)?;
            let up = gate_up.narrow(
                1,
                self.architecture.ffn_hidden,
                self.architecture.ffn_hidden,
            )?;
            let activated = if self.optional_fusion && self.device.is_cuda() && self.optional_attention_for_rows(1) {
                graph_ops::swiglu(&gate_up)?
            } else { (gate.silu()? * up)? };
            let mlp = activated.matmul(&block.down.t()?)?;
            x = (x + mlp)?;
        }
        if commit {
            cache.length += query_length;
        }
        Ok(candle_nn::ops::rms_norm(&x, &self.final_norm, 1e-6)?)
    }

    fn predict_many(&self, hidden: &Tensor, scope: OutputScope) -> Result<Vec<Prediction>> {
        self.predict_device(hidden, scope)?.to_host()
    }

    fn predict_device(&self, hidden: &Tensor, scope: OutputScope) -> Result<DevicePrediction> {
        let _device_only = self.strict_math_guard();
        let (offset, count, include_termination) = scope.range()?;
        let attributes = hidden
            .matmul(&self.embedding.narrow(0, offset, count)?.t()?)?
            .to_dtype(DType::F32)?;
        let termination = if include_termination {
            let termination = hidden
                .matmul(&self.termination_weight.t()?)?
                .broadcast_add(&self.termination_bias)?
                .to_dtype(DType::F32)?;
            Some(termination)
        } else {
            None
        };
        Ok(DevicePrediction {
            attributes,
            termination,
            attribute_offset: offset,
        })
    }
}

pub fn validate_visible(record: &[u32; 9]) -> Result<()> {
    for field in 0..9 {
        if !(OFFSETS[field]..OFFSETS[field] + BINS[field]).contains(&record[field]) {
            bail!("第 {} 个属性 token 非法：{}", field + 1, record[field]);
        }
    }
    if record[5] > OFFSETS[5] + 2 {
        bail!("画布只接受背景、矩形或椭圆图元");
    }
    Ok(())
}

pub fn region_marker(side: usize, index: usize) -> [u32; 9] {
    let cell = 512 / side;
    let row = index / side;
    let col = index % side;
    [
        ((col * cell + cell / 2) * 2 - 1) as u32,
        OFFSETS[1] + ((row * cell + cell / 2) * 2 - 1) as u32,
        OFFSETS[2],
        OFFSETS[3] + (cell * 4) as u32,
        OFFSETS[4] + (cell * 4) as u32,
        OFFSETS[5] + 3,
        REGION_MARKER,
        REGION_MARKER,
        REGION_MARKER,
    ]
}

pub fn is_marker(row: &[u32; 9]) -> bool {
    row[6..9] == [REGION_MARKER; 3]
}
