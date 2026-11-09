//! NVIDIA-only capture primitive. Inputs, outputs and state buffers belong to
//! the caller; a captured operation must not read host values or derive its
//! position from a Rust cache-length counter.
//!
//! This module deliberately does not catch a failed launch and replay the
//! ordinary path: a launch can have partially written KV. The caller must
//! discard that session and rebuild from committed CPU history first.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphShape {
    pub logical_rows: usize,
    pub guidance: usize,
    pub capacity: usize,
    pub heads: usize,
    pub head_width: usize,
}

impl GraphShape {
    pub fn checked_bytes(&self, layers: usize) -> Result<usize> {
        if self.capacity == 0
            || self.capacity > 1024
            || self.heads == 0
            || self.head_width == 0
            || layers == 0
        {
            bail!("invalid static graph geometry");
        }
        let physical_rows = crate::model::checked_batch_rows(self.logical_rows, self.guidance)?;
        // The capture kernels use U32 metadata and flattened mask indices.
        // Reject integer truncation before allocation; available memory remains
        // the normal limit and requested rows are never silently reduced.
        let mask_elements = physical_rows.checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.capacity)).context("static graph mask index overflow")?;
        u32::try_from(mask_elements).context("static graph mask exceeds U32 indexing")?;
        u32::try_from(self.head_width).context("static graph head width exceeds U32 indexing")?;
        [
            self.logical_rows,
            self.guidance,
            self.capacity,
            self.heads,
            self.head_width,
            layers,
            2,
            std::mem::size_of::<f32>(),
        ]
        .into_iter()
        .try_fold(1usize, |bytes, n| {
            bytes
                .checked_mul(n)
                .context("static graph KV byte count overflow")
        })
    }
}

#[cfg(feature = "cuda")]
pub struct NvidiaGraph {
    // Destroy the graph before dropping its captured output allocations.
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    outputs: Vec<Tensor>,
    shape: GraphShape,
    launches: u64,
    device: Device,
}

// CUDA graph calls are not internally synchronized. This wrapper exposes only
// exclusive replay, has a unique owner, and binds the CUDA context on every
// call/drop; moving that owner between threads is safe, sharing it is not.
#[cfg(feature = "cuda")]
unsafe impl Send for NvidiaGraph {}

#[cfg(feature = "cuda")]
impl Drop for NvidiaGraph {
    fn drop(&mut self) {
        if let Ok(device) = self.device.as_cuda_device() {
            let _ = device.cuda_stream().context().bind_to_thread();
        }
    }
}

#[cfg(feature = "cuda")]
impl NvidiaGraph {
    /// Warm exactly the same operation outside capture. `restore` resets only
    /// session-owned buffers, never a live caller KV. Constants and all kernel
    /// parameter shapes must have been created before the capture starts.
    pub fn capture(
        device: &Device,
        shape: GraphShape,
        mut operation: impl FnMut() -> Result<Vec<Tensor>>,
        mut restore: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        shape.checked_bytes(1)?;
        let mut checked_operation = || {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(||operation())) {
                Ok(result)=>result,
                Err(_)=>Err(anyhow::anyhow!("可选Graph组件初始化异常，已取消捕获")),
            }
        };
        let cuda = device
            .as_cuda_device()
            .context("CUDA Graph requires a CUDA device")?;
        let stream = cuda.cuda_stream();
        if !stream.context().has_async_alloc() {
            bail!(
                "CUDA graph requires capture-safe asynchronous allocation; use ordinary execution"
            );
        }
        // Every model operation, mutable graph input and readback uses this
        // one serialized worker stream. Only temporary tensors created during
        // warmup/recording are untracked; the prior context setting is restored
        // on success, error and unwinding. No external stream receives them.
        struct TrackingGuard {
            context: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
            restore: bool,
        }
        impl Drop for TrackingGuard {
            fn drop(&mut self) {
                if self.restore {
                    unsafe {
                        self.context.enable_event_tracking();
                    }
                }
            }
        }
        device.synchronize()?;
        let tracking = TrackingGuard {
            context: stream.context().clone(),
            restore: stream.context().is_event_tracking(),
        };
        if tracking.restore {
            unsafe {
                tracking.context.disable_event_tracking();
            }
        }
        // No global Candle HtoD parameter cache is enabled: it retains device
        // contexts after model replacement. The static decoder uses only
        // graph-safe custom kernels and contiguous operations / 2D cuBLAS.
        for _ in 0..2 {
            restore()?;
            let output = checked_operation()?;
            if output.is_empty() {
                bail!("cannot capture a graph without outputs");
            }
            device.synchronize()?;
        }
        restore()?;
        device.synchronize()?;
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .context("CUDA graph capture could not start")?;
        let recorded = checked_operation();
        // End capture on every path, including a failed recording. Otherwise
        // the shared worker stream would remain in capture mode.
        let ended = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        let outputs = match recorded {
            Ok(outputs) => outputs,
            Err(error) => {
                drop(ended);
                restore().context("graph recording failed and session reset also failed")?;
                return Err(error.context("CUDA graph operation recording failed"));
            }
        };
        let graph = ended
            .context("CUDA graph capture/instantiation failed")?
            .context("CUDA graph capture produced no graph")?;
        // Capturing records work but does not run it. The scratch buffers are
        // restored to the original committed state before the first replay.
        restore()?;
        device.synchronize()?;
        graph
            .upload()
            .context("CUDA graph resources could not be uploaded")?;
        drop(tracking);
        Ok(Self {
            graph,
            outputs,
            shape,
            launches: 0,
            device: device.clone(),
        })
    }

    pub fn replay(&mut self, expected: &GraphShape) -> Result<&[Tensor]> {
        if &self.shape != expected {
            bail!("captured graph shape differs from current request");
        }
        self.graph
            .launch()
            .context("CUDA graph replay failed; discard this KV session before fallback")?;
        self.launches += 1;
        Ok(&self.outputs)
    }

    pub fn launches(&self) -> u64 {
        self.launches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_geometry_is_checked_before_allocating_or_capturing() {
        let shape = GraphShape {
            logical_rows: 8,
            guidance: 2,
            capacity: 1024,
            heads: 24,
            head_width: 64,
        };
        assert_eq!(
            shape.checked_bytes(35).unwrap(),
            8 * 2 * 1024 * 24 * 64 * 35 * 2 * 4
        );
        for shape in [
            GraphShape {
                logical_rows: 0,
                ..shape.clone()
            },
            GraphShape {
                guidance: 3,
                ..shape.clone()
            },
            GraphShape {
                capacity: 1025,
                ..shape.clone()
            },
        ] {
            assert!(shape.checked_bytes(35).is_err());
        }
        assert!(shape.checked_bytes(usize::MAX).is_err());
    }

    #[test]
    fn graph_geometry_preserves_large_requested_batches_without_a_fixed_lane_cap() {
        for logical_rows in [17, 32, 64, 96, 1000] {
            let shape = GraphShape { logical_rows, guidance: 2, capacity: 1024, heads: 24, head_width: 64 };
            assert_eq!(shape.checked_bytes(35).unwrap(), logical_rows * 2 * 1024 * 24 * 64 * 35 * 2 * 4);
        }
        let shape = GraphShape { logical_rows: usize::MAX, guidance: 2, capacity: 1024, heads: 24, head_width: 64 };
        assert!(shape.checked_bytes(35).is_err());
        let shape = GraphShape { logical_rows: 1, heads: u32::MAX as usize, ..shape };
        assert!(shape.checked_bytes(35).is_err());
    }
}
