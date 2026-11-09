use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Int8KvStatus {
    pub requested: bool,
    pub available: bool,
    pub ready: bool,
    pub enabled: bool,
    pub format: String,
    pub reason: String,
    pub packed_bytes_per_token_head: usize,
    pub scale_bytes_per_token_head: usize,
}

impl Default for Int8KvStatus {
    fn default() -> Self {
        Self { requested: false, available: false, ready: false, enabled: false,
            format: "BF16/F32".into(), reason: "历史KV INT8未请求".into(),
            packed_bytes_per_token_head: 128, scale_bytes_per_token_head: 8 }
    }
}

impl PrimitiveModel {
    pub fn int8_kv_status(&self) -> Int8KvStatus { self.int8_kv.clone() }
    pub fn int8_kv_enabled(&self) -> bool { self.int8_kv.enabled }

    /// Must run before creating request caches. Compilation or execution
    /// failures are returned to the caller's existing safe-fallback classifier.
    pub fn prepare_int8_kv(&mut self, requested: bool) -> Result<Int8KvStatus> {
        self.int8_kv.requested = requested;
        self.int8_kv.enabled = false;
        self.int8_kv.available = cfg!(feature = "cuda") && self.device.is_cuda()
            && self.architecture.head_width == 64;
        self.int8_kv.format = format!("{:?}", self.dtype);
        if !requested {
            self.int8_kv.reason = "历史KV INT8已关闭，保留普通浮点缓存".into();
            return Ok(self.int8_kv_status());
        }
        if !self.int8_kv.available {
            self.int8_kv.reason = "此原生后端的历史KV INT8需要CUDA与64维注意力头，保留普通浮点缓存".into();
            return Ok(self.int8_kv_status());
        }
        if !self.int8_kv.ready {
            if let Err(error) = int8_kv_ops::prepare(&self.device) {
                self.int8_kv.reason = format!("历史KV INT8准备未完成：{error:#}");
                return Err(error);
            }
            self.int8_kv.ready = true;
        }
        self.int8_kv.enabled = true;
        self.fp32_kv_storage = false;
        self.int8_kv.format = "INT8 symmetric per-token/head64 + FP32 scales".into();
        self.int8_kv.reason = "已提交历史INT8；当前K/V保持浮点；注意力读取时融合反量化；F32 Graph关闭".into();
        Ok(self.int8_kv_status())
    }
}
