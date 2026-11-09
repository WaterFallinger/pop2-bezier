//! Capacity-triggered rolling context. A region boundary alone never reorders.
//! The caller retains authoritative raw history; this only plans model input.
use anyhow::{bail, Context, Result};

type Record = [u32; 9];
const MAX_CONTEXT: usize = 1024;

/// The candidate already includes the new region marker and any unconsumed
/// end/interrupt controls. `remaining_after_active` counts only the maximum
/// still-unwritten primitives plus the region's final end control.
/// Returning None is the sole capacity trigger for a new priority projection.
pub(super) fn reuse_or_repack_plan(
    previous: Option<&[Record]>,
    raw: &[Record],
    raw_cursor: usize,
    remaining_after_active: usize,
    text_len: usize,
) -> Result<Option<Vec<Record>>> {
    let budget = MAX_CONTEXT.checked_sub(text_len).context("文本超出1024上下文")?;
    let (old, tail) = if let Some(previous) = previous {
        let tail = raw.get(raw_cursor..).context("模型窗口的原始历史游标已失效")?;
        (previous, tail)
    } else {
        if raw_cursor != 0 { bail!("没有模型缓存时原始历史游标必须为零"); }
        (&[][..], raw)
    };
    let needed = old.len().checked_add(tail.len())
        .and_then(|n| n.checked_add(remaining_after_active))
        .context("完整区域的上下文容量计算溢出")?;
    if needed > budget { return Ok(None); }
    let mut records = Vec::new();
    records.try_reserve_exact(old.len() + tail.len()).context("无法分配模型上下文记录")?;
    records.extend_from_slice(old);
    records.extend_from_slice(tail);
    Ok(Some(records))
}

/// Reuse is by exact ordered records. The caller also fixes model, text, CFG,
/// absolute position offset and cache format for the request. If truncation
/// leaves no suffix, replay its final record to refresh stale output heads.
pub(super) fn reusable_visual_prefix(previous: &[Record], next: &[Record]) -> usize {
    let common = previous.iter().zip(next).take_while(|(a,b)| a==b).count();
    if common == next.len() && common < previous.len() {
        common.saturating_sub(1)
    } else { common }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(value:u32)->Record{[value;9]}

    #[test]
    fn coarse_and_fine_exact_fit_append_but_one_token_short_reorders(){
        // old end + new marker are already in the candidate. Reserve max
        // shapes and one end: coarse 20+1, fine 10+1, without double marker.
        for (cap,text) in [(20,2),(10,2),(20,224),(10,224)] {
            let old=vec![r(1);1024-text-(cap+1)-2];
            let mut raw=old.clone();raw.extend([r(7066),r(7057)]);
            let planned=reuse_or_repack_plan(Some(&old),&raw,old.len(),cap+1,text).unwrap().unwrap();
            assert_eq!(planned,raw);
            assert_eq!(planned.len()+cap+1+text,1024);
            raw.push(r(3));
            assert!(reuse_or_repack_plan(Some(&old),&raw,old.len(),cap+1,text).unwrap().is_none());
        }
    }

    #[test]
    fn initial_fitting_input_preserves_chronology_even_if_distance_order_differs(){
        let raw=[r(7068),r(1),r(88),r(7066),r(7057),r(22)];
        assert_eq!(reuse_or_repack_plan(None,&raw,0,11,224).unwrap().unwrap(),raw);
    }

    #[test]
    fn successive_boundaries_append_only_new_raw_records_without_reordering(){
        let mut raw=vec![r(7068),r(4),r(3),r(2)];let mut window=raw.clone();let mut cursor=raw.len();
        for index in 0..40 {
            raw.extend([r(7066),r(7057),r(100+index)]);
            window=reuse_or_repack_plan(Some(&window),&raw,cursor,11,2).unwrap().unwrap();
            cursor=raw.len();assert_eq!(window,raw);
        }
    }

    #[test]
    fn compacted_window_can_append_while_large_raw_history_stays_untouched(){
        let mut raw=vec![r(1);1500];let old=vec![r(2);900];let cursor=raw.len();
        raw.extend([r(7066),r(7057)]);
        let next=reuse_or_repack_plan(Some(&old),&raw,cursor,11,2).unwrap().unwrap();
        assert_eq!(&next[..900],old);assert_eq!(&next[900..],&raw[cursor..]);assert_eq!(raw.len(),1502);
    }

    #[test]
    fn resumed_partial_region_reserves_only_its_remaining_maximum_and_end(){
        let raw=vec![r(3);1000];
        assert!(reuse_or_repack_plan(None,&raw,0,3,20).unwrap().is_some());
        assert!(reuse_or_repack_plan(None,&raw,0,5,20).unwrap().is_none());
    }

    #[test]
    fn invalid_cursor_and_overflow_are_errors_not_silent_history_changes(){
        assert!(reuse_or_repack_plan(Some(&[r(1)]),&[],1,11,2).is_err());
        assert!(reuse_or_repack_plan(None,&[],1,11,2).is_err());
        assert!(reuse_or_repack_plan(None,&[r(1)],0,usize::MAX,2).is_err());
        assert!(reuse_or_repack_plan(None,&[],0,1,1025).is_err());
    }

    #[test]
    fn longest_prefix_respects_record_order_and_refreshes_truncated_pending(){
        let old=[r(1),r(2),r(3),r(4)];
        assert_eq!(reusable_visual_prefix(&old,&old),4);
        assert_eq!(reusable_visual_prefix(&old,&[r(1),r(2),r(4),r(3)]),2);
        assert_eq!(reusable_visual_prefix(&old,&old[..3]),2);
        assert_eq!(reusable_visual_prefix(&old,&[r(5)]),0);
        assert_eq!(reusable_visual_prefix(&old,&[]),0);
    }
}
