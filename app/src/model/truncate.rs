//! Retain an identical committed prefix without copying its device storage.
//! Subsequent appends go through the ordinary COW guards, so a branch snapshot
//! keeps its original suffix even after this cache rewinds and overwrites it.
use super::*;

impl KvCache {
    pub fn validate_truncate_committed(&self, new_len: usize) -> Result<()> {
        if new_len > self.length {
            bail!("cannot extend committed KV by truncation");
        }
        // All checks happen before mutating any layer or the host length.
        for layer in &self.layers {
            if layer.len() != self.length {
                bail!("cannot truncate an inconsistent multi-layer cache");
            }
            layer.validate_truncate_committed(new_len)?;
        }
        Ok(())
    }

    pub fn truncate_committed(&mut self, new_len: usize) -> Result<()> {
        self.validate_truncate_committed(new_len)?;
        for layer in &mut self.layers { layer.truncate_committed(new_len)?; }
        self.length = new_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(tokens: usize, base: f32) -> Tensor {
        Tensor::from_vec((0..2 * tokens * 64).map(|i| base + i as f32 * 0.007).collect::<Vec<_>>(),
            (2, tokens, 64), &Device::Cpu).unwrap()
    }
    fn cache(int8: bool) -> KvCache {
        let records = input(4, -0.75);
        let mut layers = vec![LayerCache::default(); 2];
        for layer in &mut layers {
            if int8 { layer.query_int8(&records, &records, 0, true).unwrap(); }
            else { layer.query(&records, &records, 0, true).unwrap(); }
        }
        KvCache { layers, length: 4, position_offset: 7,
            storage_dtype: if int8 { DType::U8 } else { DType::F32 } }
    }
    fn prefix(cache: &mut KvCache) -> Vec<u8> {
        if cache.int8_kv_enabled() {
            let scratch = input(1, 0.);
            let read = cache.layers[0].query_int8(&scratch, &scratch, cache.length, false).unwrap();
            read.key.narrow(1, 0, cache.length).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<u8>().unwrap()
        } else {
            cache.layers[0].key().unwrap().unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
                .into_iter().flat_map(f32::to_le_bytes).collect()
        }
    }
    #[test]
    fn rewound_float_and_int8_branches_preserve_snapshots_and_absolute_offsets() {
        for int8 in [false, true] {
            let mut live = cache(int8);
            let original = prefix(&mut live);
            let mut snapshot = live.clone();
            live.truncate_committed(2).unwrap();
            assert_eq!(live.absolute_position(), 9);
            assert_eq!(live.layer_lengths().unwrap(), [2, 2]);
            let replacement = input(1, 8.);
            for layer in &mut live.layers {
                if int8 { layer.query_int8(&replacement, &replacement, 2, true).unwrap(); }
                else { layer.query(&replacement, &replacement, 2, true).unwrap(); }
            }
            live.length += 1;
            assert_eq!(snapshot.length, 4);
            assert_eq!(prefix(&mut snapshot), original);
        }
    }
    #[test]
    fn truncate_zero_keeps_capacity_and_invalid_multilayer_truncate_is_atomic() {
        for int8 in [false, true] {
            let mut live = cache(int8);
            let bytes = live.reserved_kv_bytes();
            assert!(live.truncate_committed(5).is_err());
            assert_eq!(live.length, 4);
            live.layers[1].truncate_committed(3).unwrap();
            assert!(live.truncate_committed(2).is_err());
            assert_eq!(live.length, 4);
            assert_eq!(live.layers[0].len(), 4);
            assert_eq!(live.layers[1].len(), 3);
            live.layers[0].truncate_committed(3).unwrap(); live.length = 3;
            live.truncate_committed(0).unwrap();
            assert_eq!(live.layer_lengths().unwrap(), [0, 0]);
            assert_eq!(live.reserved_kv_bytes(), bytes);
            assert_eq!(live.absolute_position(), 7);
        }
    }
}
