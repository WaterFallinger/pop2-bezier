//! Independent data-batch, compute-dispatch and in-flight limits.
use crate::{chunked_ranges, engine::GpuEngine, types::Config};
use std::collections::VecDeque;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub struct ExecutionSettings {
    pub images_per_gpu_batch: usize,
    pub images_per_dispatch: usize,
    pub max_in_flight: usize,
}

#[derive(Debug)]
pub struct BufferCapacity {
    pub max_images_per_gpu: u64,
    pub buffer_limit_bytes: u64,
    pub pixel_buffer_bytes: u64,
}

impl ExecutionSettings {
    pub fn from_env() -> Result<Self, String> {
        let settings = Self {
            images_per_gpu_batch: crate::positive_usize_env_or(
                "SHAPE_RENDERER_IMAGES_PER_GPU_BATCH",
                crate::DEFAULT_IMAGES_PER_GPU_BATCH,
            )?,
            images_per_dispatch: crate::positive_usize_env_or(
                "SHAPE_RENDERER_IMAGES_PER_GPU_DISPATCH",
                crate::DEFAULT_IMAGES_PER_GPU_DISPATCH,
            )?,
            max_in_flight: crate::positive_usize_env_or(
                "SHAPE_RENDERER_MAX_IN_FLIGHT_SUBMISSIONS",
                crate::DEFAULT_MAX_IN_FLIGHT_SUBMISSIONS,
            )?,
        };
        settings.validate()?;
        Ok(settings)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.images_per_gpu_batch == 0
            || self.images_per_dispatch == 0
            || self.max_in_flight == 0
        {
            return Err("All three GPU scheduling settings must be positive".into());
        }
        if self.images_per_gpu_batch > (u32::MAX / crate::W / crate::H) as usize {
            return Err("GPU data batch exceeds the shader's pixel-address range".into());
        }
        Ok(())
    }

    /// Check the full resident data batch, not merely one dispatch chunk.
    /// This must run before decoding images or allocating large GPU buffers.
    pub fn validate_device_capacity(
        &self,
        limits: &wgpu::Limits,
        width: u32,
        height: u32,
        history_records: usize,
    ) -> Result<BufferCapacity, String> {
        if width == 0 || height == 0 || history_records == 0 {
            return Err(
                "Buffer capacity check requires positive dimensions and history capacity".into(),
            );
        }
        let pixel_bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(std::mem::size_of::<u32>() as u64))
            .ok_or("Pixel buffer size overflow")?;
        let history_bytes = (history_records as u64)
            .checked_mul(std::mem::size_of::<crate::types::GpuShape>() as u64)
            .ok_or("History buffer size overflow")?;
        let requirements = [
            ("目标/画布像素", pixel_bytes),
            (
                "图像状态",
                std::mem::size_of::<crate::types::ImageState>() as u64,
            ),
            ("历史记录", history_bytes),
            ("背景颜色", std::mem::size_of::<u32>() as u64),
        ];
        let buffer_limit_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let &(resource, largest_per_image) =
            requirements.iter().max_by_key(|(_, size)| *size).unwrap();
        let max_images_per_gpu = buffer_limit_bytes / largest_per_image;
        let requested = (self.images_per_gpu_batch as u64)
            .checked_mul(largest_per_image)
            .ok_or("Requested data batch buffer size overflow")?;
        if self.images_per_gpu_batch as u64 > max_images_per_gpu {
            return Err(format!(
                "每卡数据批量 {} 超过 GPU 缓冲区容量：{}缓冲区需要 {} 字节，单缓冲区/绑定上限 {} 字节；当前 {}x{} 配置最多支持 {} 张/卡。请降低 SHAPE_RENDERER_IMAGES_PER_GPU_BATCH；仅降低 SHAPE_RENDERER_IMAGES_PER_GPU_DISPATCH 不会缩小整批缓冲区。",
                self.images_per_gpu_batch, resource, requested, buffer_limit_bytes,
                width, height, max_images_per_gpu
            ));
        }
        Ok(BufferCapacity {
            max_images_per_gpu,
            buffer_limit_bytes,
            pixel_buffer_bytes: pixel_bytes * self.images_per_gpu_batch as u64,
        })
    }
}

pub struct DispatchChunk {
    // Keep each immutable uniform alive along with its bind group.
    _config: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub image_count: u32,
}

/// Share target/canvas/state/history/background across chunks. Only offset and count differ.
pub fn build_dispatches(
    engine: &GpuEngine,
    num_images: usize,
    max_images: usize,
    resources: [&wgpu::Buffer; 5],
) -> Vec<DispatchChunk> {
    assert!(num_images > 0 && max_images > 0);
    chunked_ranges(num_images, max_images)
        .into_iter()
        .map(|range| {
            let image_count = (range.end - range.start) as u32;
            let config = Config {
                image_offset: range.start as u32,
                dispatch_images: image_count,
                num_images: num_images as u32,
                pad: 0,
            };
            let buffer = engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Immutable Dispatch Config"),
                    contents: bytemuck::bytes_of(&config),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }];
            entries.extend(resources.iter().enumerate().map(|(index, resource)| {
                wgpu::BindGroupEntry {
                    binding: index as u32 + 1,
                    resource: resource.as_entire_binding(),
                }
            }));
            let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Image Sub-batch Bind Group"),
                layout: &engine.bind_group_layout,
                entries: &entries,
            });
            DispatchChunk {
                _config: buffer,
                bind_group,
                image_count,
            }
        })
        .collect()
}

/// Bounds unfinished compute submissions independently on each GPU.
pub struct SubmissionWindow {
    limit: usize,
    pending: VecDeque<wgpu::SubmissionIndex>,
}

impl SubmissionWindow {
    pub fn new(limit: usize) -> Self {
        assert!(limit > 0);
        Self {
            limit,
            pending: VecDeque::new(),
        }
    }

    pub fn submit_step(
        &mut self,
        engine: &GpuEngine,
        chunks: &[DispatchChunk],
    ) -> Result<(), String> {
        for chunk in chunks {
            let submission = engine.dispatch_step(&chunk.bind_group, chunk.image_count);
            self.pending.push_back(submission);
            if self.pending.len() >= self.limit {
                let oldest = self.pending.pop_front().unwrap();
                engine.wait_for_submission(oldest, "GPU in-flight submission limit")?;
            }
        }
        Ok(())
    }

    pub fn drain(&mut self, engine: &GpuEngine) -> Result<(), String> {
        if let Some(last) = self.pending.pop_back() {
            engine.wait_for_submission(last, "Complete submissions before readback")?;
        }
        self.pending.clear();
        Ok(())
    }

    #[cfg(test)]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_are_independent_and_reject_zero() {
        let mut s = ExecutionSettings {
            images_per_gpu_batch: 600,
            images_per_dispatch: 200,
            max_in_flight: 1,
        };
        s.validate().unwrap();
        assert_eq!(
            chunked_ranges(600, s.images_per_dispatch),
            vec![0..200, 200..400, 400..600]
        );
        assert_eq!(
            chunked_ranges(600, 128),
            vec![0..128, 128..256, 256..384, 384..512, 512..600]
        );
        s.images_per_dispatch = 1000;
        s.validate().unwrap();
        assert_eq!(chunked_ranges(600, s.images_per_dispatch), vec![0..600]);
        for field in 0..3 {
            let mut invalid = s;
            match field {
                0 => invalid.images_per_gpu_batch = 0,
                1 => invalid.images_per_dispatch = 0,
                _ => invalid.max_in_flight = 0,
            };
            assert!(invalid.validate().is_err());
        }
        assert_eq!(std::mem::size_of::<Config>(), 16);
    }

    #[test]
    fn device_capacity_checks_real_buffer_and_binding_limits() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_buffer_size = i32::MAX as u64;
        limits.max_storage_buffer_binding_size = 1 << 31;
        let mut settings = ExecutionSettings {
            images_per_gpu_batch: 2000,
            images_per_dispatch: 2000,
            max_in_flight: 1,
        };
        let capacity = settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .unwrap();
        assert_eq!(capacity.max_images_per_gpu, 2047);
        assert_eq!(capacity.pixel_buffer_bytes, 2_097_152_000);
        settings.images_per_gpu_batch = 2047;
        settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .unwrap();
        settings.images_per_gpu_batch = 2048;
        assert!(settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .is_err());
        settings.images_per_gpu_batch = 3000;
        settings.images_per_dispatch = 500;
        let error = settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .unwrap_err();
        assert!(error.contains("3145728000") && error.contains("2047"));

        // Even with a large allocation limit, binding limits independently constrain a batch.
        limits.max_buffer_size = u64::MAX;
        limits.max_storage_buffer_binding_size = 64 * 1024 * 1024;
        settings.images_per_gpu_batch = 65;
        assert!(settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .is_err());
        settings.images_per_gpu_batch = 64;
        settings
            .validate_device_capacity(&limits, 512, 512, 895)
            .unwrap();

        // At smaller canvas sizes, history can be the largest resource.
        limits.max_buffer_size = i32::MAX as u64;
        limits.max_storage_buffer_binding_size = 1 << 31;
        settings.images_per_gpu_batch = 1;
        let c = settings
            .validate_device_capacity(&limits, 64, 64, 895)
            .unwrap();
        assert_eq!(c.max_images_per_gpu, (i32::MAX as u64) / (895 * 48));
    }
}
