//! Backend-neutral geometrically reserved KV storage with copy-on-write.
//! Tensor clone/view operations themselves are aliases in Candle, so the Arc
//! below tracks cache owners independently from temporary tensor read views.

use super::{write_plan::BatchWritePlan, MAX_CONTEXT};
use anyhow::{bail, Result};
use candle_core::Tensor;
use std::sync::{Arc, Mutex, Weak};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

struct Backing {
    key: Tensor,
    value: Tensor,
    capacity: usize,
    group: Option<Arc<BatchBacking>>,
    group_row: usize,
}

/// One allocation for the physical batch. Row Backings own disjoint head
/// ranges, and this object never owns row Backings (there is no Arc cycle).
struct BatchBacking {
    key: Tensor,
    value: Tensor,
    batch: usize,
    heads: usize,
    capacity: usize,
    /// Exactly one weak registration for each physical row. A replaced row
    /// may be overwritten only after its former Backing has no cache owners,
    /// including snapshots no longer present in the current batch.
    row_owners: Mutex<Vec<Weak<Backing>>>,
    #[cfg(test)]
    copied_history_rows: AtomicUsize,
    /// Row offsets are constant for all partial attributes in one decode round.
    /// One U32 index tensor serves both K and V scatter writes.
    scatter_indices: Mutex<Option<(Vec<usize>, Tensor)>>,
}

const MIN_CAPACITY: usize = 64;

#[derive(Clone, Default)]
pub(super) struct LayerCache {
    backing: Option<Arc<Backing>>,
    committed_len: usize,
    int8: Option<Box<super::int8_cache::Int8LayerCache>>,
}

impl LayerCache {
    /// Stable allocation identity used only to validate a graph session. The
    /// graph owner pins a cache clone, preventing allocator pointer reuse.
    pub(super) fn graph_storage_identity(&self) -> Option<usize> {
        self.backing.as_ref().map(|backing| Arc::as_ptr(backing) as usize)
    }

    pub(super) fn adopt_graph_view(key: Tensor, value: Tensor, capacity: usize, committed_len: usize) -> Result<Self> {
        let (heads, actual_capacity, width) = key.dims3()?;
        if heads == 0 || width == 0 || actual_capacity != capacity
            || value.dims3()? != (heads, capacity, width)
            || key.dtype() != candle_core::DType::F32 || value.dtype() != key.dtype()
            || !key.device().same_device(value.device()) || committed_len > capacity || capacity > MAX_CONTEXT {
            bail!("invalid graph KV view");
        }
        Ok(Self { backing: Some(Arc::new(Backing { key, value, capacity, group: None, group_row: 0 })), committed_len, int8: None })
    }

    pub(super) fn advance_graph_length(&mut self, previous: usize, next: usize) -> Result<()> {
        if self.committed_len != previous || next != previous + 1
            || self.backing.as_ref().is_none_or(|backing| next > backing.capacity) {
            bail!("graph KV commit length differs from validated host state");
        }
        self.committed_len = next;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.committed_len
    }

    pub(super) fn validate_truncate_committed(&self, new_len: usize) -> Result<()> {
        if new_len > self.committed_len {
            bail!("KV metadata truncation cannot extend committed history");
        }
        if let Some(int8) = &self.int8 {
            if self.backing.is_some() || int8.len() != self.committed_len {
                bail!("INT8 outer cache length or storage differs");
            }
            int8.validate_truncate_committed(new_len)?;
        } else if self.backing.as_ref().map_or(self.committed_len != 0, |b| self.committed_len > b.capacity) {
            bail!("KV committed history has no sufficient backing");
        }
        Ok(())
    }

    /// Length-only rollback. Keep the allocation and shared ownership intact;
    /// a later write still performs COW if another snapshot owns this Backing.
    pub(super) fn truncate_committed(&mut self, new_len: usize) -> Result<()> {
        self.validate_truncate_committed(new_len)?;
        if let Some(int8) = &mut self.int8 { int8.truncate_committed(new_len)?; }
        self.committed_len = new_len;
        Ok(())
    }

    pub fn reserved_bytes(&self) -> usize {
        if let Some(int8) = &self.int8 { return int8.reserved_bytes(); }
        self.backing.as_ref().map_or(0, |b| {
            b.key.elem_count() * b.key.dtype().size_in_bytes()
                + b.value.elem_count() * b.value.dtype().size_in_bytes()
        })
    }

    pub(super) fn int8_scale_bytes(&self) -> usize { self.int8.as_ref().map_or(0, |cache| cache.scale_bytes()) }

    pub(super) fn arena(&self) -> Option<(Tensor, Tensor)> {
        self.backing.as_ref().map(|backing| (backing.key.clone(), backing.value.clone()))
    }

    pub(super) fn batch_arena(&self) -> Option<(Tensor, Tensor)> {
        let group = self.backing.as_ref()?.group.as_ref()?;
        Some((group.key.clone(), group.value.clone()))
    }

    pub(super) fn query_int8(&mut self, k: &Tensor, v: &Tensor, past: usize, commit: bool)
        -> Result<super::int8_cache::Int8Read> {
        if self.backing.is_some() || past != self.committed_len {
            bail!("INT8 history cannot reuse an existing floating cache or mismatched length");
        }
        let cache = self.int8.get_or_insert_with(Default::default);
        let result = cache.query(k, v, past, commit)?;
        self.committed_len = cache.len();
        Ok(result)
    }

    pub(super) fn query_int8_prefill(&mut self, k: &Tensor, v: &Tensor, past: usize, commit: bool)
        -> Result<(Tensor, Tensor)> {
        let prefix = self.int8.as_ref().map(|cache| cache.prefill_prefix(k.dtype())).transpose()?.flatten();
        self.query_int8(k, v, past, commit)?;
        match prefix {
            Some((key, value)) => Ok((Tensor::cat(&[&key, k], 1)?, Tensor::cat(&[&value, v], 1)?)),
            None => Ok((k.clone(), v.clone())),
        }
    }

    pub(super) fn query_int8_batch(caches: &mut [&mut Self], k: &Tensor, v: &Tensor,
        past: &[usize], commit: bool, plan: Option<&BatchWritePlan>) -> Result<super::int8_cache::Int8Read> {
        if caches.len() != past.len() || caches.iter().zip(past).any(|(cache, &length)|
            cache.backing.is_some() || cache.committed_len != length) {
            bail!("INT8 batch history cannot reuse floating caches or mismatched lengths");
        }
        let mut int8 = caches.iter_mut().map(|cache| &mut **cache.int8.get_or_insert_with(Default::default)).collect::<Vec<_>>();
        let read = super::int8_cache::Int8LayerCache::query_batch(&mut int8, k, v, past, commit, plan)?;
        drop(int8);
        for cache in caches { cache.committed_len = cache.int8.as_ref().unwrap().len(); }
        Ok(read)
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.backing.as_ref().map_or(0, |b| b.capacity)
    }

    #[cfg(test)]
    pub(super) fn batch_copied_history_rows(&self) -> Option<usize> {
        self.backing.as_ref()?.group.as_ref()
            .map(|group|group.copied_history_rows.load(Ordering::Relaxed))
    }

    /// Only the committed prefix. The returned view is generally strided, and
    /// is not a deep snapshot. Appending never overwrites that prefix.
    pub fn key(&self) -> Result<Option<Tensor>> {
        if self.int8.is_some() { bail!("INT8 history requires fused quantized attention, not an ordinary floating read"); }
        if self.committed_len == 0 {
            return Ok(None);
        }
        self.backing
            .as_ref()
            .map(|b| b.key.narrow(1, 0, self.committed_len).map_err(Into::into))
            .transpose()
    }
    pub fn value(&self) -> Result<Option<Tensor>> {
        if self.int8.is_some() { bail!("INT8 history requires fused quantized attention, not an ordinary floating read"); }
        if self.committed_len == 0 {
            return Ok(None);
        }
        self.backing
            .as_ref()
            .map(|b| b.value.narrow(1, 0, self.committed_len).map_err(Into::into))
            .transpose()
    }

    /// Write new/scratch records into [heads, capacity, head_width]. Only a
    /// complete commit advances persistent length; partial queries reuse the
    /// scratch range at exactly `past_len`. Returned scratch views may change
    /// after the next query and must not be treated as immutable snapshots.
    pub fn query(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        past_len: usize,
        commit: bool,
    ) -> Result<(Tensor, Tensor)> {
        if self.int8.is_some() { bail!("ordinary floating query cannot mutate INT8 history"); }
        if past_len != self.committed_len {
            bail!(
                "KV past_len {past_len} 与已提交长度 {} 不一致",
                self.committed_len
            );
        }
        let (heads, query_len, width) = k.dims3()?;
        if v.dims3()? != (heads, query_len, width) {
            bail!("K/V查询形状不一致");
        }
        if heads == 0 || query_len == 0 || width == 0 {
            bail!("KV查询维度必须非零");
        }
        if k.dtype() != v.dtype() || k.device().location() != v.device().location() {
            bail!("K/V查询精度或设备不一致");
        }
        let end = past_len
            .checked_add(query_len)
            .ok_or_else(|| anyhow::anyhow!("KV长度溢出"))?;
        if end > MAX_CONTEXT {
            bail!("KV容量为{MAX_CONTEXT}，查询需要{end}；必须先重建上下文");
        }
        if let Some(backing) = &self.backing {
            if backing.key.dims3()? != (heads, backing.capacity, width)
                || backing.value.dims3()? != (heads, backing.capacity, width)
                || backing.key.dtype() != k.dtype()
                || backing.value.dtype() != v.dtype()
                || backing.key.device().location() != k.device().location()
                || backing.value.device().location() != v.device().location()
            {
                bail!("KV缓存形状、精度或设备与新查询不一致");
            }
        }
        let current_capacity = self.backing.as_ref().map_or(0, |b| b.capacity);
        let mut capacity = current_capacity.max(MIN_CAPACITY.min(MAX_CONTEXT));
        while capacity < end {
            capacity = (capacity * 2).min(MAX_CONTEXT);
        }
        let allocate = || -> Result<Backing> {
            Ok(Backing {
                key: Tensor::zeros((heads, capacity, width), k.dtype(), k.device())?,
                value: Tensor::zeros((heads, capacity, width), v.dtype(), v.device())?,
                capacity,
                group: None,
                group_row: 0,
            })
        };
        if self.backing.is_none() {
            self.backing = Some(Arc::new(allocate()?));
        } else if capacity > current_capacity
            || Arc::strong_count(self.backing.as_ref().unwrap()) > 1
        {
            let old = self.backing.as_ref().unwrap();
            let independent = allocate()?;
            if self.committed_len > 0 {
                // A head-major prefix narrow is not generally contiguous.
                // Growth and COW share this one copy. Copy committed values
                // only, never the whole reserve or another owner's scratch.
                let key = old.key.narrow(1, 0, self.committed_len)?.contiguous()?;
                let value = old.value.narrow(1, 0, self.committed_len)?.contiguous()?;
                independent.key.slice_set(&key, 1, 0)?;
                independent.value.slice_set(&value, 1, 0)?;
            }
            self.backing = Some(Arc::new(independent));
        }
        let backing = self.backing.as_ref().unwrap();
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // slice_set dispatches to the backend's in-place copy2d. It copies the
        // query span only, with no full-prefix cat or functional slice_assign.
        backing.key.slice_set(&k, 1, past_len)?;
        backing.value.slice_set(&v, 1, past_len)?;
        if commit {
            self.committed_len = end;
        }
        Ok((
            backing.key.narrow(1, 0, end)?,
            backing.value.narrow(1, 0, end)?,
        ))
    }

    /// Decode one position per physical row, directly in a shared batch arena.
    /// Inputs are [B,H,1,D]; returned views are [B*H,max(past_len)+1,D].
    /// A shorter row's tail is attention padding, never committed history, and
    /// the caller must mask it. Returned scratch views are not snapshots.
    /// Shape/device/length validation completes before live buffers are written.
    /// A backend execution failure still requires discarding the failed KV.
    pub fn query_batch(
        caches: &mut [&mut LayerCache],
        k: &Tensor,
        v: &Tensor,
        past_lens: &[usize],
        commit: bool,
    ) -> Result<(Tensor, Tensor)> {
        Self::query_batch_with_plan(caches, k, v, past_lens, commit, None)
    }

    /// An optional immutable plan can supply the same scatter geometry to all
    /// layers. Validation happens before any live KV write or handle change.
    pub fn query_batch_with_plan(
        caches: &mut [&mut LayerCache],
        k: &Tensor,
        v: &Tensor,
        past_lens: &[usize],
        commit: bool,
        plan: Option<&BatchWritePlan>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, heads, queries, width) = k.dims4()?;
        if caches.iter().any(|cache| cache.int8.is_some()) { bail!("ordinary floating batch cannot mutate INT8 history"); }
        if batch == 0 || heads == 0 || queries != 1 || width == 0 {
            bail!("batch KV requires nonempty [B,H,1,D]");
        }
        if caches.len() != batch || past_lens.len() != batch {
            bail!("batch KV rows and cache/length counts differ");
        }
        if v.dims4()? != (batch, heads, queries, width)
            || k.dtype() != v.dtype()
            || k.device().location() != v.device().location()
        {
            bail!("batch K/V shape, dtype or device mismatch");
        }
        if let Some(plan) = plan {
            plan.validate(past_lens, heads, width, k.device())?;
        }
        let mut capacity = MIN_CAPACITY.min(MAX_CONTEXT);
        let mut max_end = 0;
        for (cache, &past_len) in caches.iter().zip(past_lens) {
            if past_len != cache.committed_len {
                bail!("batch KV supplied past length differs from committed history");
            }
            let end = past_len
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("KV length overflow"))?;
            if end > MAX_CONTEXT {
                bail!("batch KV needs context rebuild");
            }
            max_end = max_end.max(end);
            if let Some(backing) = &cache.backing {
                if backing.key.dims3()? != (heads, backing.capacity, width)
                    || backing.value.dims3()? != (heads, backing.capacity, width)
                    || backing.key.dtype() != k.dtype()
                    || backing.value.dtype() != v.dtype()
                    || backing.key.device().location() != k.device().location()
                    || backing.value.device().location() != v.device().location()
                    || past_len > backing.capacity
                {
                    bail!("batch KV existing row is incompatible");
                }
                capacity = capacity.max(backing.capacity);
            } else if past_len != 0 {
                bail!("batch KV committed row has no storage");
            }
        }
        while capacity < max_end {
            capacity = (capacity * 2).min(MAX_CONTEXT);
        }

        let (group, replace_rows) = if let Some(reusable) =
            reusable_batch_group(caches, batch, heads, capacity, width, k)? {
            reusable
        } else {
            let group = Arc::new(BatchBacking {
                key: Tensor::zeros((batch * heads, capacity, width), k.dtype(), k.device())?,
                value: Tensor::zeros((batch * heads, capacity, width), v.dtype(), v.device())?,
                batch,
                heads,
                capacity,
                row_owners: Mutex::new((0..batch).map(|_| Weak::new()).collect()),
                #[cfg(test)]
                copied_history_rows: AtomicUsize::new(0),
                scatter_indices: Mutex::new(None),
            });
            (group, (0..batch).collect())
        };
        // Sources normally are QKV outputs. If a diagnostic caller passed a
        // view into this very arena, snapshot that source before in-place writes.
        let normalized_k = k.contiguous()?;
        let normalized_v = v.contiguous()?;
        let normalized_k = if shares_storage(&normalized_k, &group.key)
            || shares_storage(&normalized_k, &group.value)
        {
            normalized_k.copy()?
        } else {
            normalized_k
        };
        let normalized_v = if shares_storage(&normalized_v, &group.key)
            || shares_storage(&normalized_v, &group.value)
        {
            normalized_v.copy()?
        } else {
            normalized_v
        };
        // Normalize/snapshot query sources BEFORE replacing any old slot: an
        // input Tensor view can outlive its former cache owner and alias it.
        let mut replacements = Vec::with_capacity(replace_rows.len());
        for row in replace_rows {
            let cache = &caches[row];
            let key = group.key.narrow(0, row * heads, heads)?;
            let value = group.value.narrow(0, row * heads, heads)?;
            if cache.committed_len > 0 {
                let old = cache.backing.as_ref().unwrap();
                let old_key = old.key.narrow(1, 0, cache.committed_len)?.contiguous()?;
                let old_value = old.value.narrow(1, 0, cache.committed_len)?.contiguous()?;
                key.slice_set(&old_key, 1, 0)?;
                value.slice_set(&old_value, 1, 0)?;
                #[cfg(test)]
                group.copied_history_rows.fetch_add(1, Ordering::Relaxed);
            }
            replacements.push((row, Arc::new(Backing {
                key, value, capacity,
                group: Some(group.clone()), group_row: row,
            })));
        }
        if past_lens.iter().all(|&length| length == past_lens[0]) {
            group.key.slice_set(
                &normalized_k.reshape((batch * heads, 1, width))?,
                1,
                past_lens[0],
            )?;
            group.value.slice_set(
                &normalized_v.reshape((batch * heads, 1, width))?,
                1,
                past_lens[0],
            )?;
        } else {
            let indices = if let Some(plan) = plan {
                plan.indices().clone()
            } else {
                let mut cached = group
                    .scatter_indices
                    .lock()
                    .map_err(|_| anyhow::anyhow!("batch KV scatter index cache poisoned"))?;
                if let Some((_, indices)) =
                    cached.as_ref().filter(|(offsets, _)| offsets == past_lens)
                {
                    indices.clone()
                } else {
                    // Each (physical head, width element) has a unique target.
                    // Only its sequence coordinate differs across ragged rows.
                    let offsets = past_lens
                        .iter()
                        .flat_map(|&offset| std::iter::repeat(offset as u32).take(heads * width))
                        .collect::<Vec<_>>();
                    let indices = Tensor::from_vec(offsets, (batch * heads, 1, width), k.device())?;
                    *cached = Some((past_lens.to_vec(), indices.clone()));
                    indices
                }
            };
            group.key.scatter_set(
                &indices,
                &normalized_k.reshape((batch * heads, 1, width))?,
                1,
            )?;
            group.value.scatter_set(
                &indices,
                &normalized_v.reshape((batch * heads, 1, width))?,
                1,
            )?;
        }
        if !replacements.is_empty() {
            let mut owners = group.row_owners.lock()
                .map_err(|_| anyhow::anyhow!("batch KV row owner registry poisoned"))?;
            for (row, backing) in replacements {
                owners[row] = Arc::downgrade(&backing);
                caches[row].backing = Some(backing);
            }
        }
        if commit {
            for (cache, &past_len) in caches.iter_mut().zip(past_lens) {
                cache.committed_len = past_len + 1;
            }
        }
        Ok((
            group.key.narrow(1, 0, max_end)?,
            group.value.narrow(1, 0, max_end)?,
        ))
    }
}

/// Prove row ownership, not merely ownership of the outer batch Arc. Tensor
/// reads are aliases/scratch views, whereas a LayerCache snapshot owns its
/// Backing and therefore keeps the registered weak slot alive.
fn reusable_batch_group(
    caches: &[&mut LayerCache], batch: usize, heads: usize, capacity: usize,
    width: usize, input: &Tensor,
) -> Result<Option<(Arc<BatchBacking>, Vec<usize>)>> {
    for candidate in caches.iter().filter_map(|cache| cache.backing.as_ref()?.group.as_ref()) {
        if candidate.batch != batch || candidate.heads != heads || candidate.capacity != capacity
            || candidate.key.dims() != [batch * heads, capacity, width]
            || candidate.value.dims() != candidate.key.dims()
            || candidate.key.dtype() != input.dtype() || candidate.value.dtype() != input.dtype()
            || !candidate.key.device().same_device(input.device()) || !candidate.value.device().same_device(input.device()) {
            return Ok(None);
        }
        let owners = candidate.row_owners.lock()
            .map_err(|_| anyhow::anyhow!("batch KV row owner registry poisoned"))?;
        if owners.len() != batch { return Ok(None); }
        let mut changed = Vec::new();
        let mut safe = true;
        for (row, cache) in caches.iter().enumerate() {
            if let Some(backing) = &cache.backing {
                if let Some(group) = &backing.group {
                    // Permuted rows or another arena are deliberately excluded.
                    if !Arc::ptr_eq(group, candidate) || backing.group_row != row
                        || owners[row].as_ptr() != Arc::as_ptr(backing)
                        || Arc::strong_count(backing) != 1 {
                        safe = false; break;
                    }
                    continue;
                }
                // Only a genuinely independent, same-capacity prefill row can
                // replace an orphaned slot. Graph/adopted Tensor aliases cannot.
                if backing.capacity != capacity
                    || [&backing.key, &backing.value].iter().any(|source|
                        shares_storage(source, &candidate.key) || shares_storage(source, &candidate.value)) {
                    safe = false; break;
                }
            }
            // A snapshot of a removed row is not in `caches`, but still owns
            // its registered Backing. Never overwrite that snapshot's range.
            if owners[row].upgrade().is_some() { safe = false; break; }
            changed.push(row);
        }
        // Mixed/permuted groups are intentionally unsupported, so another
        // candidate cannot make an unsafe first candidate valid. Do not repeat
        // this same ownership scan once for every remaining row.
        return Ok(safe.then(||(candidate.clone(), changed)));
    }
    Ok(None)
}

fn shares_storage(a: &Tensor, b: &Tensor) -> bool {
    let a_pointer = {
        let (storage, _) = a.storage_and_layout();
        &*storage as *const _
    };
    let b_pointer = {
        let (storage, _) = b.storage_and_layout();
        &*storage as *const _
    };
    std::ptr::eq(a_pointer, b_pointer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Immutable cat oracle retained independently of the new in-place path.
    fn cat_reference(old: Option<&Tensor>, current: &Tensor) -> Tensor {
        match old {
            Some(old) => Tensor::cat(&[old, current], 1).unwrap(),
            None => current.clone(),
        }
    }
    fn data(heads: usize, len: usize, width: usize, base: f32) -> Tensor {
        Tensor::from_vec(
            (0..heads * len * width)
                .map(|i| base + i as f32 * 0.25)
                .collect::<Vec<_>>(),
            (heads, len, width),
            &Device::Cpu,
        )
        .unwrap()
    }
    fn flat(t: &Tensor) -> Vec<f32> {
        t.contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn batch_data(batch: usize, heads: usize, width: usize, base: f32) -> Tensor {
        Tensor::stack(
            &(0..batch)
                .map(|row| data(heads, 1, width, base + row as f32 * 100.))
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap()
    }

    fn grouped_rows(dtype: DType) -> Vec<LayerCache> {
        let mut caches = vec![LayerCache::default(); 3];
        for (row, cache) in caches.iter_mut().enumerate() {
            let prefix = data(2, 3, 3, row as f32 * 10.).to_dtype(dtype).unwrap();
            cache.query(&prefix, &prefix, 0, true).unwrap();
        }
        let current = batch_data(3, 2, 3, 100.).to_dtype(dtype).unwrap();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &current, &current, &[3; 3], false).unwrap();
        drop(refs);
        caches
    }

    #[test]
    fn independent_reprefill_rejoins_its_orphan_slot_without_copying_other_rows() {
        for dtype in [DType::BF16, DType::F32] {
            for changed in 0..3 {
                let mut caches = grouped_rows(dtype);
                let old_group = caches[(changed + 1) % 3].backing.as_ref().unwrap().group.clone().unwrap();
                let copied_before = old_group.copied_history_rows.load(Ordering::Relaxed);
                let owner_ids = caches.iter().map(|c|Arc::as_ptr(c.backing.as_ref().unwrap())).collect::<Vec<_>>();
                // Plain read views are scratch aliases, not snapshot owners.
                let read_views = caches.iter().map(|c|c.key().unwrap().unwrap()).collect::<Vec<_>>();
                caches[changed] = LayerCache::default();
                let rebuilt = data(2, 2, 3, 900.).to_dtype(dtype).unwrap();
                caches[changed].query(&rebuilt, &rebuilt, 0, true).unwrap();
                let lengths = (0..3).map(|r|if r==changed{2}else{3}).collect::<Vec<_>>();
                let current = batch_data(3, 2, 3, 500.).to_dtype(dtype).unwrap();
                let mut refs = caches.iter_mut().collect::<Vec<_>>();
                let (keys, _) = LayerCache::query_batch(&mut refs, &current, &current, &lengths, true).unwrap();
                drop(refs);
                assert_eq!(old_group.key.id(), caches[0].backing.as_ref().unwrap().group.as_ref().unwrap().key.id());
                assert_eq!(old_group.copied_history_rows.load(Ordering::Relaxed)-copied_before,1,
                    "only the independently rebuilt row may copy historical K/V");
                for row in 0..3 {
                    let before = if row==changed { rebuilt.clone() } else { read_views[row].clone() };
                    let tail = current.narrow(0,row,1).unwrap().reshape((2,1,3)).unwrap();
                    let actual = keys.narrow(0,row*2,2).unwrap().narrow(1,0,lengths[row]+1).unwrap();
                    assert_eq!(flat(&actual.to_dtype(DType::F32).unwrap()),flat(&cat_reference(Some(&before),&tail).to_dtype(DType::F32).unwrap()));
                    if row!=changed { assert_eq!(Arc::as_ptr(caches[row].backing.as_ref().unwrap()),owner_ids[row]); }
                }
            }
        }
    }

    #[test]
    fn detached_snapshot_of_replaced_slot_forces_new_batch_arena() {
        let mut caches = grouped_rows(DType::F32);
        let mut snapshot = caches[1].clone();
        let saved = flat(&snapshot.key().unwrap().unwrap());
        let old_group = caches[0].backing.as_ref().unwrap().group.clone().unwrap();
        caches[1] = LayerCache::default();
        let rebuilt = data(2,2,3,900.);
        caches[1].query(&rebuilt,&rebuilt,0,true).unwrap();
        let current = batch_data(3,2,3,500.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs,&current,&current,&[3,2,3],true).unwrap();
        drop(refs);
        assert_ne!(old_group.key.id(),caches[0].backing.as_ref().unwrap().group.as_ref().unwrap().key.id());
        assert_eq!(flat(&snapshot.key().unwrap().unwrap()),saved);
        snapshot.query(&data(2,1,3,-80.),&data(2,1,3,-70.),3,true).unwrap();
        assert_eq!(flat(&snapshot.key().unwrap().unwrap().narrow(1,0,3).unwrap()),saved);
    }

    #[test]
    fn empty_replacement_row_reuses_orphan_slot_without_history_copy() {
        let mut caches=grouped_rows(DType::F32);
        let group=caches[1].backing.as_ref().unwrap().group.clone().unwrap();
        let copied=group.copied_history_rows.load(Ordering::Relaxed);
        caches[0]=LayerCache::default();let current=batch_data(3,2,3,700.);
        let mut refs=caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs,&current,&current,&[0,3,3],true).unwrap();drop(refs);
        assert_eq!(group.key.id(),caches[0].backing.as_ref().unwrap().group.as_ref().unwrap().key.id());
        assert_eq!(group.copied_history_rows.load(Ordering::Relaxed),copied);
        assert_eq!(flat(&caches[0].key().unwrap().unwrap()),flat(&current.narrow(0,0,1).unwrap().reshape((2,1,3)).unwrap()));
    }

    #[test]
    fn permutation_shared_owner_capacity_growth_and_adopted_alias_keep_safe_fallback() {
        for case in 0..4 {
            let mut caches = grouped_rows(DType::F32);
            let old_group = caches[0].backing.as_ref().unwrap().group.clone().unwrap();
            let snapshot = (case==1).then(||caches[2].clone());
            let saved = snapshot.as_ref().map(|s|flat(&s.key().unwrap().unwrap()));
            let mut lengths = vec![3;3];
            match case {
                0 => caches.swap(0,1),
                1 => {},
                2 => { caches[1]=LayerCache::default();let p=data(2,65,3,800.);caches[1].query(&p,&p,0,true).unwrap();lengths[1]=65; },
                _ => {caches[1]=LayerCache::adopt_graph_view(old_group.key.narrow(0,2,2).unwrap(),old_group.value.narrow(0,2,2).unwrap(),64,3).unwrap();},
            }
            let current=batch_data(3,2,3,500.);let mut refs=caches.iter_mut().collect::<Vec<_>>();
            LayerCache::query_batch(&mut refs,&current,&current,&lengths,true).unwrap();drop(refs);
            assert_ne!(old_group.key.id(),caches[0].backing.as_ref().unwrap().group.as_ref().unwrap().key.id(),"unsafe reuse for case {case}");
            if let Some(snapshot)=snapshot {assert_eq!(flat(&snapshot.key().unwrap().unwrap()),saved.unwrap());}
        }
    }

    #[test]
    fn truncate_is_metadata_only_and_append_preserves_longer_snapshots() {
        let mut cache=grouped_rows(DType::F32).remove(0);
        let snapshot=cache.clone();let saved=flat(&snapshot.key().unwrap().unwrap());
        let owner=cache.graph_storage_identity();let bytes=cache.reserved_bytes();
        assert!(cache.truncate_committed(4).is_err());assert_eq!(cache.len(),3);
        cache.truncate_committed(1).unwrap();
        assert_eq!(cache.graph_storage_identity(),owner);assert_eq!(cache.reserved_bytes(),bytes);
        let next=data(2,1,3,900.);cache.query(&next,&next,1,true).unwrap();
        assert_ne!(cache.graph_storage_identity(),owner);assert_eq!(snapshot.len(),3);
        assert_eq!(flat(&snapshot.key().unwrap().unwrap()),saved);
        let owner=cache.graph_storage_identity();cache.truncate_committed(0).unwrap();
        assert_eq!(cache.graph_storage_identity(),owner);assert_eq!(cache.reserved_bytes(),bytes);assert!(cache.key().unwrap().is_none());
    }

    #[test]
    fn one_write_plan_is_reused_by_thirty_five_layers_without_local_index_caches() {
        let offsets = [1, 3];
        let plan = BatchWritePlan::new(&offsets, 2, 3, &Device::Cpu).unwrap();
        let index_id = plan.indices().id();
        let mut layers = (0..35)
            .map(|layer| {
                let mut rows = vec![LayerCache::default(); 2];
                for (row, cache) in rows.iter_mut().enumerate() {
                    let k = data(2, offsets[row], 3, layer as f32 * 1000. + row as f32 * 100.);
                    let v = data(
                        2,
                        offsets[row],
                        3,
                        layer as f32 * 1000. + row as f32 * 100. + 50.,
                    );
                    cache.query(&k, &v, 0, true).unwrap();
                }
                rows
            })
            .collect::<Vec<_>>();
        for (layer, caches) in layers.iter_mut().enumerate() {
            let old_k = caches
                .iter()
                .map(|c| c.key().unwrap().unwrap())
                .collect::<Vec<_>>();
            let old_v = caches
                .iter()
                .map(|c| c.value().unwrap().unwrap())
                .collect::<Vec<_>>();
            let k = batch_data(2, 2, 3, layer as f32 * 1000. + 500.);
            let v = batch_data(2, 2, 3, layer as f32 * 1000. + 700.);
            let cloned_plan = plan.clone();
            let mut refs = caches.iter_mut().collect::<Vec<_>>();
            let (keys, values) = LayerCache::query_batch_with_plan(
                &mut refs,
                &k,
                &v,
                &offsets,
                false,
                Some(&cloned_plan),
            )
            .unwrap();
            drop(refs);
            assert_eq!(cloned_plan.indices().id(), index_id);
            for row in 0..2 {
                let current_k = k.narrow(0, row, 1).unwrap().reshape((2, 1, 3)).unwrap();
                let current_v = v.narrow(0, row, 1).unwrap().reshape((2, 1, 3)).unwrap();
                assert_eq!(
                    flat(
                        &keys
                            .narrow(0, row * 2, 2)
                            .unwrap()
                            .narrow(1, 0, offsets[row] + 1)
                            .unwrap()
                    ),
                    flat(&cat_reference(Some(&old_k[row]), &current_k)),
                );
                assert_eq!(
                    flat(
                        &values
                            .narrow(0, row * 2, 2)
                            .unwrap()
                            .narrow(1, 0, offsets[row] + 1)
                            .unwrap()
                    ),
                    flat(&cat_reference(Some(&old_v[row]), &current_v)),
                );
                assert_eq!(caches[row].len(), offsets[row]);
                assert_eq!(
                    flat(&caches[row].key().unwrap().unwrap()),
                    flat(&old_k[row])
                );
                assert_eq!(
                    flat(&caches[row].value().unwrap().unwrap()),
                    flat(&old_v[row])
                );
            }
            let group = caches[0].backing.as_ref().unwrap().group.as_ref().unwrap();
            assert!(group.scatter_indices.lock().unwrap().is_none());
        }
    }

    #[test]
    fn wrong_shared_write_plan_is_rejected_before_history_or_scratch_is_touched() {
        for offsets in [[1, 3], [1, 1]] {
            let mut caches = vec![LayerCache::default(); 2];
            for (row, cache) in caches.iter_mut().enumerate() {
                let k = data(2, offsets[row], 3, row as f32 * 100.);
                let v = data(2, offsets[row], 3, row as f32 * 100. + 50.);
                cache.query(&k, &v, 0, true).unwrap();
            }
            let current = batch_data(2, 2, 3, 500.);
            let valid = BatchWritePlan::new(&offsets, 2, 3, &Device::Cpu).unwrap();
            let mut refs = caches.iter_mut().collect::<Vec<_>>();
            LayerCache::query_batch_with_plan(
                &mut refs,
                &current,
                &current,
                &offsets,
                false,
                Some(&valid),
            )
            .unwrap();
            drop(refs);
            let group = caches[0]
                .backing
                .as_ref()
                .unwrap()
                .group
                .as_ref()
                .unwrap()
                .clone();
            let before_k = flat(&group.key);
            let before_v = flat(&group.value);
            let before_owners = caches
                .iter()
                .map(|c| Arc::as_ptr(c.backing.as_ref().unwrap()))
                .collect::<Vec<_>>();
            let wrong_plans = [
                BatchWritePlan::new(&[0, offsets[1]], 2, 3, &Device::Cpu).unwrap(),
                BatchWritePlan::new(&offsets, 1, 3, &Device::Cpu).unwrap(),
                BatchWritePlan::new(&offsets, 2, 2, &Device::Cpu).unwrap(),
            ];
            for wrong in &wrong_plans {
                let mut refs = caches.iter_mut().collect::<Vec<_>>();
                assert!(LayerCache::query_batch_with_plan(
                    &mut refs,
                    &current,
                    &current,
                    &offsets,
                    true,
                    Some(wrong),
                )
                .is_err());
                drop(refs);
                assert_eq!(flat(&group.key), before_k);
                assert_eq!(flat(&group.value), before_v);
                for row in 0..2 {
                    assert_eq!(caches[row].len(), offsets[row]);
                    assert_eq!(
                        Arc::as_ptr(caches[row].backing.as_ref().unwrap()),
                        before_owners[row]
                    );
                }
            }
        }
    }

    #[test]
    fn committed_offsets_require_a_fresh_shared_plan_and_preserve_old_plan() {
        let mut caches = vec![LayerCache::default(); 2];
        for (row, cache) in caches.iter_mut().enumerate() {
            let prefix = data(1, row + 1, 2, row as f32 * 10.);
            cache.query(&prefix, &prefix, 0, true).unwrap();
        }
        let initial = BatchWritePlan::new(&[1, 2], 1, 2, &Device::Cpu).unwrap();
        let initial_ids = initial
            .indices()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let current = batch_data(2, 1, 2, 50.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch_with_plan(
            &mut refs,
            &current,
            &current,
            &[1, 2],
            true,
            Some(&initial),
        )
        .unwrap();
        drop(refs);
        let committed = caches
            .iter()
            .map(|c| flat(&c.key().unwrap().unwrap()))
            .collect::<Vec<_>>();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        assert!(LayerCache::query_batch_with_plan(
            &mut refs,
            &current,
            &current,
            &[2, 3],
            true,
            Some(&initial)
        )
        .is_err());
        drop(refs);
        for row in 0..2 {
            assert_eq!(flat(&caches[row].key().unwrap().unwrap()), committed[row]);
        }
        let refreshed = BatchWritePlan::new(&[2, 3], 1, 2, &Device::Cpu).unwrap();
        assert_ne!(refreshed.indices().id(), initial.indices().id());
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch_with_plan(
            &mut refs,
            &current,
            &current,
            &[2, 3],
            false,
            Some(&refreshed),
        )
        .unwrap();
        drop(refs);
        assert_eq!(
            initial
                .indices()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            initial_ids
        );
        assert_eq!(
            caches.iter().map(LayerCache::len).collect::<Vec<_>>(),
            [2, 3]
        );
        for row in 0..2 {
            assert_eq!(flat(&caches[row].key().unwrap().unwrap()), committed[row]);
        }
    }

    #[test]
    fn ragged_scatter_reuses_indices_across_partials_and_refreshes_after_commit() {
        let mut caches = vec![LayerCache::default(); 2];
        let mut expected_k = Vec::new();
        let mut expected_v = Vec::new();
        for (row, cache) in caches.iter_mut().enumerate() {
            let count = if row == 0 { 1 } else { 3 };
            let k = data(2, count, 3, 10. + row as f32 * 20.);
            let v = data(2, count, 3, 110. + row as f32 * 20.);
            cache.query(&k, &v, 0, true).unwrap();
            expected_k.push(k);
            expected_v.push(v);
        }
        let mut previous_index: Option<Tensor> = None;
        let mut group: Option<Arc<BatchBacking>> = None;
        for (step, commit) in [false, false, true, false].into_iter().enumerate() {
            let offsets = if step < 3 { vec![1, 3] } else { vec![2, 4] };
            let k = batch_data(2, 2, 3, 500. + step as f32 * 100.);
            let v = batch_data(2, 2, 3, 1500. + step as f32 * 100.);
            let mut refs = caches.iter_mut().collect::<Vec<_>>();
            let (keys, values) =
                LayerCache::query_batch(&mut refs, &k, &v, &offsets, commit).unwrap();
            drop(refs);
            let arena = caches[0].backing.as_ref().unwrap().group.clone().unwrap();
            if let Some(old) = &group {
                assert!(Arc::ptr_eq(old, &arena));
            }
            let index = arena
                .scatter_indices
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .1
                .clone();
            let ids = index.flatten_all().unwrap().to_vec1::<u32>().unwrap();
            let expected_ids = offsets
                .iter()
                .flat_map(|&p| std::iter::repeat(p as u32).take(6))
                .collect::<Vec<_>>();
            assert_eq!(ids, expected_ids);
            if let Some(old) = &previous_index {
                if step < 3 {
                    assert_eq!(index.id(), old.id());
                } else {
                    assert_ne!(index.id(), old.id());
                }
            }
            for row in 0..2 {
                let current_k = k.narrow(0, row, 1).unwrap().reshape((2, 1, 3)).unwrap();
                let current_v = v.narrow(0, row, 1).unwrap().reshape((2, 1, 3)).unwrap();
                let oracle_k = cat_reference(Some(&expected_k[row]), &current_k);
                let oracle_v = cat_reference(Some(&expected_v[row]), &current_v);
                assert_eq!(
                    flat(
                        &keys
                            .narrow(0, row * 2, 2)
                            .unwrap()
                            .narrow(1, 0, offsets[row] + 1)
                            .unwrap()
                    ),
                    flat(&oracle_k)
                );
                assert_eq!(
                    flat(
                        &values
                            .narrow(0, row * 2, 2)
                            .unwrap()
                            .narrow(1, 0, offsets[row] + 1)
                            .unwrap()
                    ),
                    flat(&oracle_v)
                );
                if commit {
                    expected_k[row] = oracle_k;
                    expected_v[row] = oracle_v;
                }
                assert_eq!(
                    flat(&caches[row].key().unwrap().unwrap()),
                    flat(&expected_k[row])
                );
                assert_eq!(
                    flat(&caches[row].value().unwrap().unwrap()),
                    flat(&expected_v[row])
                );
            }
            previous_index = Some(index);
            group = Some(arena);
        }
    }

    #[test]
    fn ragged_scatter_keeps_f16_bf16_and_f32_data_types() {
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let mut caches = vec![LayerCache::default(); 2];
            for (row, cache) in caches.iter_mut().enumerate() {
                let prefix = data(1, row + 1, 2, row as f32 + 1.)
                    .to_dtype(dtype)
                    .unwrap();
                cache.query(&prefix, &prefix, 0, true).unwrap();
            }
            let current = batch_data(2, 1, 2, 4.).to_dtype(dtype).unwrap();
            let before = caches
                .iter()
                .map(|c| c.key().unwrap().unwrap().copy().unwrap())
                .collect::<Vec<_>>();
            let mut refs = caches.iter_mut().collect::<Vec<_>>();
            let (keys, values) =
                LayerCache::query_batch(&mut refs, &current, &current, &[1, 2], true).unwrap();
            drop(refs);
            assert_eq!(keys.dtype(), dtype);
            assert_eq!(values.dtype(), dtype);
            for row in 0..2 {
                let input = current
                    .narrow(0, row, 1)
                    .unwrap()
                    .reshape((1, 1, 2))
                    .unwrap();
                let expected = cat_reference(Some(&before[row]), &input);
                let actual = keys
                    .narrow(0, row, 1)
                    .unwrap()
                    .narrow(1, 0, row + 2)
                    .unwrap();
                assert_eq!(
                    flat(&actual.to_dtype(DType::F32).unwrap()),
                    flat(&expected.to_dtype(DType::F32).unwrap())
                );
            }
        }
    }

    #[test]
    fn batch_arena_matches_each_cat_oracle_and_reuses_one_storage() {
        let mut caches = vec![LayerCache::default(); 3];
        let mut expected = Vec::new();
        for (row, cache) in caches.iter_mut().enumerate() {
            let prefix = data(2, 3, 4, row as f32 * 10.);
            cache.query(&prefix, &prefix, 0, true).unwrap();
            expected.push(prefix);
        }
        let first = batch_data(3, 2, 4, 500.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let (keys, values) =
            LayerCache::query_batch(&mut refs, &first, &first, &[3, 3, 3], false).unwrap();
        drop(refs);
        assert_eq!(keys.dims(), &[6, 4, 4]);
        assert!(!keys.is_contiguous());
        for row in 0..3 {
            let current = first.narrow(0, row, 1).unwrap().reshape((2, 1, 4)).unwrap();
            let oracle = cat_reference(Some(&expected[row]), &current);
            assert_eq!(flat(&keys.narrow(0, row * 2, 2).unwrap()), flat(&oracle));
            assert_eq!(flat(&values.narrow(0, row * 2, 2).unwrap()), flat(&oracle));
            assert_eq!(caches[row].len(), 3);
            assert_eq!(
                flat(&caches[row].key().unwrap().unwrap()),
                flat(&expected[row])
            );
        }
        let group = caches[0].backing.as_ref().unwrap().group.clone().unwrap();
        let second = batch_data(3, 2, 4, 900.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let (keys, _) =
            LayerCache::query_batch(&mut refs, &second, &second, &[3, 3, 3], true).unwrap();
        drop(refs);
        assert!(Arc::ptr_eq(
            &group,
            caches[2].backing.as_ref().unwrap().group.as_ref().unwrap()
        ));
        for row in 0..3 {
            let current = second
                .narrow(0, row, 1)
                .unwrap()
                .reshape((2, 1, 4))
                .unwrap();
            assert_eq!(
                flat(&keys.narrow(0, row * 2, 2).unwrap()),
                flat(&cat_reference(Some(&expected[row]), &current))
            );
            assert_eq!(caches[row].len(), 4);
        }
    }

    #[test]
    fn ragged_batch_has_maskable_tails_and_invalid_last_row_writes_nothing() {
        let lengths = [0, 2, 5];
        let mut caches = vec![LayerCache::default(); 3];
        let mut old = Vec::new();
        for (row, cache) in caches.iter_mut().enumerate() {
            if lengths[row] > 0 {
                let prefix = data(2, lengths[row], 3, row as f32 * 10.);
                cache.query(&prefix, &prefix, 0, true).unwrap();
                old.push(Some(prefix));
            } else {
                old.push(None);
            }
        }
        let current = batch_data(3, 2, 3, 600.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let (keys, _) =
            LayerCache::query_batch(&mut refs, &current, &current, &lengths, false).unwrap();
        drop(refs);
        assert_eq!(keys.dims(), &[6, 6, 3]);
        for row in 0..3 {
            let source = current
                .narrow(0, row, 1)
                .unwrap()
                .reshape((2, 1, 3))
                .unwrap();
            let visible = keys
                .narrow(0, row * 2, 2)
                .unwrap()
                .narrow(1, 0, lengths[row] + 1)
                .unwrap();
            assert_eq!(
                flat(&visible),
                flat(&cat_reference(old[row].as_ref(), &source))
            );
            assert_eq!(caches[row].len(), lengths[row]);
            assert_eq!(caches[row].key().unwrap().is_none(), lengths[row] == 0);
        }
        let before = flat(&keys);
        let other = batch_data(3, 2, 3, 800.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        assert!(LayerCache::query_batch(&mut refs, &other, &other, &[0, 2, 4], true).is_err());
        drop(refs);
        assert_eq!(flat(&keys), before);
        assert_eq!(
            caches.iter().map(LayerCache::len).collect::<Vec<_>>(),
            lengths
        );
    }

    #[test]
    fn cloned_batch_detaches_before_partial_and_commit() {
        let mut original = vec![LayerCache::default(); 2];
        let first = batch_data(2, 2, 3, 10.);
        let mut refs = original.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &first, &first, &[0, 0], true).unwrap();
        drop(refs);
        let before = original
            .iter()
            .map(|c| flat(&c.key().unwrap().unwrap()))
            .collect::<Vec<_>>();
        let mut branch = original.clone();
        let other = batch_data(2, 2, 3, 700.);
        let mut refs = branch.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &other, &other, &[1, 1], false).unwrap();
        drop(refs);
        assert!(!Arc::ptr_eq(
            original[0]
                .backing
                .as_ref()
                .unwrap()
                .group
                .as_ref()
                .unwrap(),
            branch[0].backing.as_ref().unwrap().group.as_ref().unwrap()
        ));
        let mut refs = branch.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &other, &other, &[1, 1], true).unwrap();
        drop(refs);
        for row in 0..2 {
            assert_eq!(original[row].len(), 1);
            assert_eq!(branch[row].len(), 2);
            assert_eq!(flat(&original[row].key().unwrap().unwrap()), before[row]);
        }
    }

    #[test]
    fn batch_growth_and_single_row_detach_can_rejoin_without_losing_prefix() {
        let mut caches = vec![LayerCache::default(); 2];
        for (row, cache) in caches.iter_mut().enumerate() {
            let prefix = data(2, 63, 3, row as f32 * 100.);
            cache.query(&prefix, &prefix, 0, true).unwrap();
        }
        let current = batch_data(2, 2, 3, 700.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &current, &current, &[63, 63], true).unwrap();
        drop(refs);
        assert!(caches.iter().all(|c| c.capacity() == 64));
        let old = caches.clone();
        let next = data(2, 1, 3, 2000.);
        caches[0].query(&next, &next, 64, true).unwrap();
        assert!(caches[0].backing.as_ref().unwrap().group.is_none());
        assert_eq!(caches[0].capacity(), 128);
        assert_eq!(old[0].len(), 64);
        let before = caches
            .iter()
            .map(|c| c.key().unwrap().unwrap().copy().unwrap())
            .collect::<Vec<_>>();
        let tail = batch_data(2, 2, 3, 3000.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let (keys, _) = LayerCache::query_batch(&mut refs, &tail, &tail, &[65, 64], true).unwrap();
        drop(refs);
        for row in 0..2 {
            assert_eq!(caches[row].capacity(), 128);
            let expected = cat_reference(
                Some(&before[row]),
                &tail.narrow(0, row, 1).unwrap().reshape((2, 1, 3)).unwrap(),
            );
            assert_eq!(
                flat(
                    &keys
                        .narrow(0, row * 2, 2)
                        .unwrap()
                        .narrow(1, 0, expected.dim(1).unwrap())
                        .unwrap()
                ),
                flat(&expected)
            );
            assert_eq!(old[row].len(), 64);
        }
    }

    #[test]
    fn aliased_input_is_snapshotted_before_batch_write() {
        let mut caches = vec![LayerCache::default()];
        let first = batch_data(1, 1, 3, 10.);
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_batch(&mut refs, &first, &first, &[0], true).unwrap();
        drop(refs);
        let source = caches[0]
            .backing
            .as_ref()
            .unwrap()
            .key
            .narrow(1, 0, 1)
            .unwrap()
            .reshape((1, 1, 1, 3))
            .unwrap();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let (keys, _) = LayerCache::query_batch(&mut refs, &source, &source, &[1], true).unwrap();
        drop(refs);
        assert_eq!(flat(&keys), vec![10., 10.25, 10.5, 10., 10.25, 10.5]);
    }

    #[test]
    fn preallocated_commit_matches_dynamic_cat_oracle() {
        let mut cache = LayerCache::default();
        let mut old_k = None;
        let mut old_v = None;
        for (step, n) in [3, 1, 5, 2].into_iter().enumerate() {
            let k = data(3, n, 4, step as f32 * 10.);
            let v = data(3, n, 4, 100. + step as f32 * 10.);
            let expected_k = cat_reference(old_k.as_ref(), &k);
            let expected_v = cat_reference(old_v.as_ref(), &v);
            let old_len = cache.len();
            let (actual_k, actual_v) = cache.query(&k, &v, old_len, true).unwrap();
            assert_eq!(flat(&actual_k), flat(&expected_k));
            assert_eq!(flat(&actual_v), flat(&expected_v));
            assert_eq!(cache.len(), old_len + n);
            assert_eq!(flat(&cache.key().unwrap().unwrap()), flat(&expected_k));
            old_k = Some(expected_k);
            old_v = Some(expected_v);
        }
    }

    #[test]
    fn capacity_grows_by_powers_of_two_without_losing_committed_prefix() {
        let mut cache = LayerCache::default();
        assert_eq!(cache.capacity(), 0);
        let mut old_k = None;
        let mut old_v = None;
        let chunks = [1, 63, 1, 63, 1, 127, 1, 255, 1, 511];
        let capacities = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024];
        for (step, (&count, &capacity)) in chunks.iter().zip(&capacities).enumerate() {
            let k = data(2, count, 3, step as f32 * 1000.);
            let v = data(2, count, 3, step as f32 * 1000. + 500.);
            let expected_k = cat_reference(old_k.as_ref(), &k);
            let expected_v = cat_reference(old_v.as_ref(), &v);
            let previous = cache.len();
            let (actual_k, actual_v) = cache.query(&k, &v, previous, true).unwrap();
            assert_eq!(cache.capacity(), capacity);
            assert_eq!(cache.len(), previous + count);
            assert_eq!(flat(&actual_k), flat(&expected_k));
            assert_eq!(flat(&actual_v), flat(&expected_v));
            old_k = Some(expected_k);
            old_v = Some(expected_v);
        }
        assert_eq!(cache.len(), MAX_CONTEXT);
    }

    #[test]
    fn cow_retains_current_reserve_and_growth_copies_no_uncommitted_scratch() {
        let mut original = LayerCache::default();
        let prefix = data(2, 70, 3, 1.);
        original.query(&prefix, &prefix, 0, true).unwrap();
        assert_eq!(original.capacity(), 128);
        let scratch = data(2, 10, 3, 9000.);
        original.query(&scratch, &scratch, 70, false).unwrap();
        let mut branch = original.clone();
        let one = data(2, 1, 3, 55.);
        branch.query(&one, &one, 70, false).unwrap();
        assert_eq!(branch.capacity(), 128);
        assert_eq!(branch.len(), 70);
        assert_eq!(flat(&branch.key().unwrap().unwrap()), flat(&prefix));
        assert_eq!(
            flat(
                &branch
                    .backing
                    .as_ref()
                    .unwrap()
                    .key
                    .narrow(1, 72, 1)
                    .unwrap()
            ),
            vec![0.; 6]
        );
        assert_ne!(
            flat(
                &original
                    .backing
                    .as_ref()
                    .unwrap()
                    .key
                    .narrow(1, 72, 1)
                    .unwrap()
            ),
            vec![0.; 6]
        );
        let mut growing = original.clone();
        let next = data(2, 80, 3, 200.);
        growing.query(&next, &next, 70, false).unwrap();
        assert_eq!(growing.capacity(), 256);
        assert_eq!(growing.len(), 70);
        assert_eq!(original.capacity(), 128);
        assert_eq!(original.len(), 70);
        assert_eq!(flat(&growing.key().unwrap().unwrap()), flat(&prefix));
        let (committed, _) = growing.query(&next, &next, 70, true).unwrap();
        assert_eq!(growing.capacity(), 256);
        assert_eq!(growing.len(), 150);
        assert_eq!(flat(&committed), flat(&cat_reference(Some(&prefix), &next)));
        assert_eq!(flat(&original.key().unwrap().unwrap()), flat(&prefix));
    }

    #[test]
    fn partial_overwrites_scratch_without_committing_or_touching_prefix() {
        let mut cache = LayerCache::default();
        let k = data(2, 4, 3, 1.);
        let v = data(2, 4, 3, 20.);
        cache.query(&k, &v, 0, true).unwrap();
        let prefix = flat(&cache.key().unwrap().unwrap());
        let first = data(2, 1, 3, 100.);
        let second = data(2, 1, 3, 200.);
        let (first_view, _) = cache.query(&first, &first, 4, false).unwrap();
        let first_snapshot = first_view.copy().unwrap();
        let (second_view, _) = cache.query(&second, &second, 4, false).unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(flat(&cache.key().unwrap().unwrap()), prefix);
        assert_ne!(flat(&first_snapshot), flat(&second_view));
        assert_eq!(flat(&second_view), flat(&cat_reference(Some(&k), &second)));
        let (committed, _) = cache.query(&second, &second, 4, true).unwrap();
        assert_eq!(cache.len(), 5);
        assert_eq!(flat(&committed), flat(&cat_reference(Some(&k), &second)));
    }

    #[test]
    fn cloned_cache_detaches_before_partial_or_commit_and_does_not_copy_scratch() {
        let mut original = LayerCache::default();
        let prefix = data(2, 3, 4, 1.);
        original.query(&prefix, &prefix, 0, true).unwrap();
        original
            .query(&data(2, 1, 4, 999.), &data(2, 1, 4, 999.), 3, false)
            .unwrap();
        let mut branch = original.clone();
        assert!(Arc::ptr_eq(
            original.backing.as_ref().unwrap(),
            branch.backing.as_ref().unwrap()
        ));
        let query = data(2, 1, 4, 80.);
        branch.query(&query, &query, 3, false).unwrap();
        assert!(!Arc::ptr_eq(
            original.backing.as_ref().unwrap(),
            branch.backing.as_ref().unwrap()
        ));
        assert_eq!(original.len(), 3);
        assert_eq!(branch.len(), 3);
        assert_eq!(flat(&original.key().unwrap().unwrap()), flat(&prefix));
        assert_eq!(
            flat(
                &original
                    .backing
                    .as_ref()
                    .unwrap()
                    .key
                    .narrow(1, 3, 1)
                    .unwrap()
            ),
            flat(&data(2, 1, 4, 999.))
        );
        assert_eq!(
            flat(
                &branch
                    .backing
                    .as_ref()
                    .unwrap()
                    .key
                    .narrow(1, 4, 1)
                    .unwrap()
            ),
            vec![0.; 8]
        );
        branch.query(&query, &query, 3, true).unwrap();
        assert_eq!(original.len(), 3);
        assert_eq!(branch.len(), 4);
        assert_eq!(flat(&original.key().unwrap().unwrap()), flat(&prefix));
    }

    #[test]
    fn default_clones_allocate_separately_and_validate_boundary_and_lengths() {
        let mut first = LayerCache::default();
        let mut second = first.clone();
        let a = data(1, 1, 2, 1.);
        let b = data(1, 1, 2, 50.);
        first.query(&a, &a, 0, true).unwrap();
        second.query(&b, &b, 0, true).unwrap();
        assert_ne!(
            flat(&first.key().unwrap().unwrap()),
            flat(&second.key().unwrap().unwrap())
        );
        assert!(first.query(&a, &a, 0, false).is_err());
        assert_eq!(first.len(), 1);
        assert!(first
            .query(&data(2, 1, 2, 0.), &data(2, 1, 2, 0.), 1, true)
            .is_err());
        assert!(first
            .query(&a, &a.to_dtype(DType::F64).unwrap(), 1, true)
            .is_err());
        let mut full = LayerCache::default();
        let block = data(1, MAX_CONTEXT, 1, 0.);
        full.query(&block, &block, 0, true).unwrap();
        assert_eq!(full.len(), MAX_CONTEXT);
        assert!(full
            .query(&data(1, 1, 1, 2.), &data(1, 1, 1, 2.), MAX_CONTEXT, true)
            .is_err());
        assert_eq!(full.len(), MAX_CONTEXT);
    }

    #[test]
    fn strided_query_source_is_normalized_and_partial_only_cache_has_no_valid_prefix() {
        let source = data(2, 4, 3, 7.).narrow(1, 1, 2).unwrap();
        assert!(!source.is_contiguous());
        let mut cache = LayerCache::default();
        let (k, _) = cache.query(&source, &source, 0, false).unwrap();
        assert_eq!(flat(&k), flat(&source));
        assert_eq!(cache.len(), 0);
        assert!(cache.key().unwrap().is_none());
        assert!(cache.value().unwrap().is_none());
        let mut clone = cache.clone();
        clone
            .query(&data(2, 1, 3, 1.), &data(2, 1, 3, 2.), 0, true)
            .unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(clone.len(), 1);
    }
}
