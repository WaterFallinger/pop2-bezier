//! Batch dimensions are resource-limited, not capped by a product preference.
use anyhow::{bail, Context, Result};

pub(crate) fn checked_batch_rows(logical_rows: usize, guidance: usize) -> Result<usize> {
    if logical_rows == 0 || !matches!(guidance, 1 | 2) {
        bail!("并行序列数必须大于零，CFG分支数必须为1或2");
    }
    let physical_rows = logical_rows
        .checked_mul(guidance)
        .context("并行序列数与CFG分支数相乘溢出")?;
    u32::try_from(physical_rows).context("并行序列的物理行数超过设备U32索引范围")?;
    // Device metadata indexes nine attributes per logical sequence with U32.
    let fields = logical_rows.checked_mul(9).context("并行属性索引数量溢出")?;
    u32::try_from(fields).context("并行属性数量超过设备U32索引范围")?;
    Ok(physical_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rows_preserve_requested_counts_above_sixteen() {
        for logical in [1, 16, 17, 32, 64, 96, 1000] {
            assert_eq!(checked_batch_rows(logical, 1).unwrap(), logical);
            assert_eq!(checked_batch_rows(logical, 2).unwrap(), logical * 2);
        }
    }

    #[test]
    fn batch_rows_reject_only_invalid_or_unrepresentable_geometry() {
        assert!(checked_batch_rows(0, 1).is_err());
        assert!(checked_batch_rows(1, 3).is_err());
        assert!(checked_batch_rows(usize::MAX, 2).is_err());
        let last_indexable = u32::MAX as usize / 9;
        assert_eq!(checked_batch_rows(last_indexable, 2).unwrap(), last_indexable * 2);
        assert!(checked_batch_rows(last_indexable + 1, 1).is_err());
    }
}
