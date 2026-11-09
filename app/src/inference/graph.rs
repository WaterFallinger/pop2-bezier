//! Optional whole-primitive CUDA graph execution. Ordinary tensor/GPU sampling
//! remains available on preparation failure. No host RNG is consumed here.
use super::device::{DeviceFrame, DeviceLaneSpec};
use super::*;
use crate::model::graph_decode::{graph_finite, graph_pack, GraphDecodeState};
use crate::model::BINS;
use candle_core::{DType, Tensor};
#[path = "graph_capture.rs"]
mod graph_capture;
use graph_capture::GraphShape;
#[cfg(feature = "cuda")]
use graph_capture::NvidiaGraph;

pub(super) enum PrimitiveGraphSession {
    #[cfg(feature = "cuda")]
    Ready(ReadySession),
    Disabled(String),
}

#[cfg(feature = "cuda")]
pub(super) struct ReadySession {
    graph: Option<NvidiaGraph>,
    shape: GraphShape,
    state: GraphDecodeState,
    alpha_bits: u32,
    cfg_bits: u32,
    bounds: Vec<Tensor>,
    field_indices: Tensor,
    partial: Tensor,
    records: Tensor,
    uniforms: Tensor,
    ones: Tensor,
    pending_attributes: Tensor,
    pending_terms: Tensor,
    next_attributes: Tensor,
    next_terms: Tensor,
    packed: Tensor,
    specs: Vec<(bool, Option<(usize, usize)>)>,
    captures: u64,
    replays: u64,
    refreshes: u64,
}

#[cfg(feature = "cuda")]
impl ReadySession {
    fn capacity(caches: &[&KvCache], specs: &[DeviceLaneSpec]) -> Result<usize> {
        let length = caches
            .iter()
            .map(|cache| cache.length)
            .max()
            .context("empty graph cache rows")?;
        let reserve = specs
            .iter()
            .map(|spec| match spec.region {
                None => 64,
                Some((4, _)) => 20,
                _ => 10,
            })
            .max()
            .unwrap_or(10);
        Ok((length.saturating_add(reserve).min(MAX_CONTEXT).div_ceil(64) * 64).min(MAX_CONTEXT))
    }
    fn new(
        engine: &NativeEngine,
        specs: &[DeviceLaneSpec],
        pending: &[&model::DevicePrediction],
        caches: &[&KvCache],
        settings: &InferenceSettings,
    ) -> Result<Self> {
        if engine.reject_graph_preparation.load(Ordering::Relaxed) {
            bail!("diagnostic graph preparation rejection before private buffers are allocated");
        }
        let batch = specs.len();
        let guidance = if (settings.cfg_scale - 1.).abs() > 1e-12 {
            2
        } else {
            1
        };
        let model = &engine.model;
        let device = &model.device;
        let capacity = Self::capacity(caches, specs)?;
        let shape = GraphShape {
            logical_rows: batch,
            guidance,
            capacity,
            heads: model.architecture.heads,
            head_width: model.architecture.head_width,
        };
        let bytes = shape.checked_bytes(model.architecture.layers)?;
        let free = device
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(anyhow::Error::msg)?
            .0;
        if bytes.saturating_add(256 * 1024 * 1024) > free {
            bail!("insufficient dedicated memory for a separate graph KV arena");
        }
        let map = (0..batch)
            .flat_map(|row| std::iter::repeat(row as u32).take(guidance))
            .collect::<Vec<_>>();
        let map = Tensor::from_vec(map, batch * guidance, device)?;
        let state = GraphDecodeState::new(model, caches, &map, capacity)?;
        let partial = Tensor::from_vec(
            (0..batch).flat_map(|_| PARTIAL).collect::<Vec<_>>(),
            (batch, 9),
            device,
        )?;
        let records = Tensor::zeros((batch, 9), DType::U32, device)?;
        let field_indices = Tensor::from_vec(
            (0..9)
                .flat_map(|field| std::iter::repeat(field as u32).take(batch))
                .collect::<Vec<_>>(),
            (9, batch, 1),
            device,
        )?;
        let mut bounds = Vec::with_capacity(9);
        for field in 0..9 {
            let mut values = Vec::with_capacity(batch * 2);
            for spec in specs {
                let candidates = field_candidates(field, spec.background, spec.region)?;
                values.extend([candidates[0] - OFFSETS[field], candidates.len() as u32]);
            }
            bounds.push(Tensor::from_vec(values, (batch, 2), device)?);
        }
        let physical = batch * guidance;
        let mut session = Self {
            graph: None,
            shape,
            state,
            alpha_bits: (settings.renyi_alpha as f32).to_bits(),
            cfg_bits: (settings.cfg_scale as f32).to_bits(),
            bounds,
            field_indices,
            partial,
            records,
            uniforms: Tensor::full(0.5f32, (9, batch), device)?,
            ones: Tensor::ones((batch, 1), DType::U32, device)?,
            pending_attributes: Tensor::zeros((physical, BINS[0] as usize), DType::F32, device)?,
            pending_terms: Tensor::zeros((physical, 2), DType::F32, device)?,
            next_attributes: Tensor::zeros((physical, BINS[0] as usize), DType::F32, device)?,
            next_terms: Tensor::zeros((physical, 2), DType::F32, device)?,
            packed: Tensor::zeros((batch, 10 + guidance * 2), DType::F32, device)?,
            specs: specs.iter().map(|s| (s.background, s.region)).collect(),
            captures: 1,
            replays: 0,
            refreshes: 0,
        };
        session.copy_pending(pending)?;
        let graph = NvidiaGraph::capture(
            device,
            session.shape.clone(),
            || session.enqueue(engine, settings),
            || Ok(()),
        )?;
        session.graph = Some(graph);
        Ok(session)
    }

    fn compatible(
        &self,
        engine: &NativeEngine,
        specs: &[DeviceLaneSpec],
        caches: &[&KvCache],
        settings: &InferenceSettings,
    ) -> bool {
        let guidance = if (settings.cfg_scale - 1.).abs() > 1e-12 {
            2
        } else {
            1
        };
        self.shape.logical_rows == specs.len()
            && self.shape.guidance == guidance
            && self.alpha_bits == (settings.renyi_alpha as f32).to_bits()
            && self.cfg_bits == (settings.cfg_scale as f32).to_bits()
            && self.shape.heads == engine.model.architecture.heads
            && caches.len() == specs.len() * guidance
            // The initial reserve chooses an allocation, not a sliding
            // recapture threshold. Keep this graph while its next slot fits.
            && caches.iter().all(|cache| {
                cache.length < self.shape.capacity && cache.absolute_position() < MAX_CONTEXT
            })
            && self.state.validate_caches(caches).is_ok()
    }

    fn copy_pending(&self, pending: &[&model::DevicePrediction]) -> Result<()> {
        let prediction = model::DevicePrediction::concat_rows(pending)?;
        if prediction.attribute_offset != 0
            || prediction.attributes.dims() != self.pending_attributes.dims()
        {
            bail!("graph pending input is not a complete next-x head");
        }
        if prediction.attributes.id() != self.pending_attributes.id() {
            self.pending_attributes
                .slice_set(&prediction.attributes, 0, 0)?;
        }
        let terms = prediction
            .termination
            .as_ref()
            .context("graph pending termination absent")?;
        if terms.id() != self.pending_terms.id() {
            self.pending_terms.slice_set(terms, 0, 0)?;
        }
        Ok(())
    }

    fn refresh(
        &mut self,
        engine: &NativeEngine,
        specs: &[DeviceLaneSpec],
        pending: &[&model::DevicePrediction],
        caches: &[&KvCache],
    ) -> Result<()> {
        let key = specs
            .iter()
            .map(|spec| (spec.background, spec.region))
            .collect::<Vec<_>>();
        if self.state.validate_caches(caches).is_err() {
            self.state.refresh(&engine.model, caches)?;
            self.refreshes += 1;
        }
        if key != self.specs {
            for field in 0..9 {
                let mut values = Vec::with_capacity(specs.len() * 2);
                for spec in specs {
                    let candidates = field_candidates(field, spec.background, spec.region)?;
                    values.extend([candidates[0] - OFFSETS[field], candidates.len() as u32]);
                }
                let source = Tensor::from_vec(values, (specs.len(), 2), &engine.model.device)?;
                self.bounds[field].slice_set(&source, 0, 0)?;
            }
            self.specs = key;
        }
        self.state.set_positions(caches, &engine.model)?;
        self.copy_pending(pending)
    }

    /// Entire captured primitive. Every output allocation returned from this
    /// closure was created before capture; temporary async allocations/free
    /// nodes are balanced before the closure returns.
    fn enqueue(&self, engine: &NativeEngine, settings: &InferenceSettings) -> Result<Vec<Tensor>> {
        let batch = self.shape.logical_rows;
        let guidance = self.shape.guidance;
        self.records.slice_set(&self.partial, 0, 0)?;
        let metadata = self.state.metadata(&engine.model)?;
        let mut prediction = model::DevicePrediction {
            attributes: self.pending_attributes.clone(),
            termination: None,
            attribute_offset: 0,
        };
        let mut valid = self.ones.clone();
        for field in 0..9 {
            if field > 0 {
                prediction = self.state.forward(
                    &engine.model,
                    &self.records,
                    &metadata,
                    model::OutputScope::Field(field),
                )?;
            }
            let uniforms = self.uniforms.narrow(0, field, 1)?.reshape((batch, 1))?;
            let (sampled, safe, _) = super::fused_sampling::sample(
                &prediction.attributes,
                &uniforms,
                &self.bounds[field],
                settings.renyi_alpha as f32,
                settings.cfg_scale as f32,
                guidance,
                OFFSETS[field],
            )?;
            valid = valid.broadcast_mul(&safe)?;
            let column = self
                .field_indices
                .narrow(0, field, 1)?
                .reshape((batch, 1))?;
            self.records.scatter_set(&column, &sampled, 1)?;
        }
        let next = self.state.forward(
            &engine.model,
            &self.records,
            &metadata,
            model::OutputScope::NextPrimitive,
        )?;
        valid = valid.broadcast_mul(&graph_finite(&next.attributes, guidance)?)?;
        let terms = next
            .termination
            .context("captured next termination absent")?;
        self.next_attributes.slice_set(&next.attributes, 0, 0)?;
        self.next_terms.slice_set(&terms, 0, 0)?;
        let packed = graph_pack(&self.records, &terms, &valid, guidance)?;
        self.packed.slice_set(&packed, 0, 0)?;
        Ok(vec![self.packed.clone()])
    }

    fn prepare_inputs(
        &mut self,
        engine: &NativeEngine,
        specs: &[DeviceLaneSpec],
        pending: &[&model::DevicePrediction],
        caches: &[&KvCache],
        uniforms: &[f32],
    ) -> Result<()> {
        self.refresh(engine, specs, pending, caches)?;
        let source = Tensor::from_vec(
            uniforms.to_vec(),
            (9, self.shape.logical_rows),
            &engine.model.device,
        )?;
        self.uniforms.slice_set(&source, 0, 0)?;
        Ok(())
    }

    fn replay(
        &mut self,
        engine: &NativeEngine,
        specs: &[DeviceLaneSpec],
        caches: &mut [&mut KvCache],
        stop: &AtomicBool,
    ) -> Result<Option<DeviceFrame>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        self.graph
            .as_mut()
            .context("CUDA graph absent after successful preparation")?
            .replay(&self.shape)?;
        self.replays += 1;
        if engine.stop_after_device_enqueue {
            stop.store(true, Ordering::Relaxed);
        }
        let host = self.packed.to_vec2::<f32>()?;
        let mut records = Vec::with_capacity(specs.len());
        let mut terms = Vec::with_capacity(specs.len());
        for (row, spec) in host.iter().zip(specs) {
            if row.len() != 10 + self.shape.guidance * 2
                || row[9 + self.shape.guidance * 2] != 1.
                || row.iter().any(|value| !value.is_finite())
            {
                bail!("CUDA graph produced nonfinite/invalid output; current primitive was not committed");
            }
            let mut record = [0u32; 9];
            for field in 0..9 {
                if row[field] < 0. || row[field].fract() != 0. || row[field] > BOS as f32 {
                    bail!("CUDA graph produced invalid token");
                }
                record[field] = row[field] as u32;
                let candidates = field_candidates(field, spec.background, spec.region)?;
                if record[field] < candidates[0] || record[field] > *candidates.last().unwrap() {
                    bail!("CUDA graph sampled outside the legal region");
                }
            }
            validate_visible(&record)?;
            records.push(record);
            terms.push(
                (0..self.shape.guidance)
                    .map(|g| [row[9 + g * 2], row[10 + g * 2]])
                    .collect::<Vec<_>>(),
            );
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // Candidate heads are separate from committed pending input. Failed
        // launches never replace the caller's previous pending prediction.
        self.pending_attributes
            .slice_set(&self.next_attributes, 0, 0)?;
        self.pending_terms.slice_set(&self.next_terms, 0, 0)?;
        self.state.commit(&engine.model, caches)?;
        Ok(Some(DeviceFrame {
            records,
            next: model::DevicePrediction {
                attributes: self.pending_attributes.clone(),
                termination: Some(self.pending_terms.clone()),
                attribute_offset: 0,
            },
            terms,
            queries: 9,
        }))
    }
}

impl NativeEngine {
    /// Preparation errors can fall back using the same uniforms and untouched
    /// ordinary cache. Execution/validation errors propagate and never resample.
    pub(super) fn try_primitive_graph_frame(
        &self,
        specs: &[DeviceLaneSpec],
        pending: &[&model::DevicePrediction],
        caches: &mut [&mut KvCache],
        settings: &InferenceSettings,
        uniforms: &[f32],
        stop: &AtomicBool,
    ) -> Result<Option<DeviceFrame>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (specs, pending, caches, settings, uniforms);
            return Ok(None);
        }
        #[cfg(feature = "cuda")]
        {
            if specs.iter().any(|spec| spec.background) || !self.model.device.is_cuda() {
                return Ok(None);
            }
            let mut slot = self
                .primitive_graph
                .lock()
                .map_err(|_| anyhow::anyhow!("primitive graph cache poisoned"))?;
            if matches!(slot.as_ref(), Some(PrimitiveGraphSession::Disabled(_))) {
                return Ok(None);
            }
            let refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            let compatible = matches!(slot.as_ref(),Some(PrimitiveGraphSession::Ready(session)) if session.compatible(self,specs,&refs,settings));
            if !compatible {
                // Release a retired graph before allocating the replacement.
                *slot = None;
                match ReadySession::new(self, specs, pending, &refs, settings) {
                    Ok(session) => {
                        self.primitive_graph_captures
                            .fetch_add(1, Ordering::Relaxed);
                        *slot = Some(PrimitiveGraphSession::Ready(session));
                    }
                    Err(error) => {
                        let reason = format!("{error:#}");
                        if graph_device_fault(&reason) {
                            self.poison_acceleration(reason.clone());
                            return Err(error);
                        }
                        self.acceleration_notice(format!(
                            "整图元Graph准备不可用，保留普通GPU执行：{reason}"
                        ));
                        *slot = Some(PrimitiveGraphSession::Disabled(reason));
                        return Ok(None);
                    }
                }
            }
            if let Some(PrimitiveGraphSession::Ready(session)) = slot.as_mut() {
                if let Err(error) = session.prepare_inputs(self, specs, pending, &refs, uniforms) {
                    let reason = format!("{error:#}");
                    if graph_device_fault(&reason) {
                        self.poison_acceleration(reason.clone());
                        return Err(error);
                    }
                    self.acceleration_notice(format!(
                        "整图元Graph输入准备失败，保留普通GPU执行：{reason}"
                    ));
                    *slot = Some(PrimitiveGraphSession::Disabled(reason));
                    return Ok(None);
                }
                let before = session.replays;
                let result = session.replay(self, specs, caches, stop);
                let delta = session.replays - before;
                let previous = self
                    .primitive_graph_replays
                    .fetch_add(delta, Ordering::Relaxed);
                if matches!(&result, Ok(Some(_))) && previous == 0 && delta > 0 {
                    self.acceleration_notice("整图元CUDA Graph已实际执行，后续图元复用当前计算图");
                }
                if let Err(error) = &result {
                    self.poison_acceleration(format!("整图元Graph执行失败：{error:#}"));
                }
                result.map_err(|error| error.context("整图元Graph执行失败，需要重新加载模型"))
            } else {
                Ok(None)
            }
        }
    }

    pub fn primitive_graph_status(&self) -> String {
        let (captures, replays) = self.primitive_graph_counters();
        let Ok(slot) = self.primitive_graph.lock() else {
            return "CUDA graph state unavailable".into();
        };
        match slot.as_ref() {
            #[cfg(feature = "cuda")]
            Some(PrimitiveGraphSession::Ready(session)) => format!(
                "CUDA Graph: captures={}, replays={}, context refreshes={}",
                captures, replays, session.refreshes
            ),
            Some(PrimitiveGraphSession::Disabled(reason)) => {
                format!("CUDA Graph fallback: {reason} (captures={captures}, replays={replays})")
            }
            None => format!("CUDA Graph not prepared (captures={captures}, replays={replays})"),
        }
    }
    pub fn primitive_graph_counters(&self) -> (u64, u64) {
        (
            self.primitive_graph_captures.load(Ordering::Relaxed),
            self.primitive_graph_replays.load(Ordering::Relaxed),
        )
    }
}

fn graph_device_fault(reason: &str) -> bool {
    super::uncertain_device_execution(reason)
}
