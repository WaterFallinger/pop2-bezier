//! Time-driven UI preview and one immutable seed range per accepted request.
//! Previewing never reserves a value or mutates a creative state. Reserving a
//! whole parallel range prevents rapid successive requests from sharing lanes.
use std::{sync::atomic::{AtomicU64,Ordering},time::{SystemTime,UNIX_EPOCH}};

static LAST_RESERVED:AtomicU64=AtomicU64::new(0);

fn clock_seed()->u64 {
    // Microseconds remain exactly representable by the numeric widget's f64
    // conversion today, unlike nanoseconds. StdRng expands this integer seed;
    // the clock is not used as the sampling distribution itself.
    SystemTime::now().duration_since(UNIX_EPOCH).ok()
        .and_then(|elapsed|u64::try_from(elapsed.as_micros()).ok()).unwrap_or(1)
}

pub(crate) fn preview_seed()->u64 {
    preview_at(&LAST_RESERVED,clock_seed())
}
fn preview_at(last:&AtomicU64,now:u64)->u64 {now.max(last.load(Ordering::Relaxed).saturating_add(1))}

pub(crate) fn reserve_seed(lanes:usize)->Result<u64,String> {
    reserve_at(&LAST_RESERVED,clock_seed(),lanes)
}

fn reserve_at(last:&AtomicU64,now:u64,lanes:usize)->Result<u64,String> {
    if lanes==0{return Err("并行推理数必须至少为1".into());}
    let span=u64::try_from(lanes).map_err(|_|"并行随机种子数量无法表示")?;
    let mut previous=last.load(Ordering::Relaxed);
    loop {
        let next=previous.checked_add(1).ok_or("自动随机种子范围已用尽，请改用手动种子")?;
        let seed=now.max(next);
        let end=seed.checked_add(span-1).ok_or("并行数超过可用随机种子范围，请减少并行数")?;
        match last.compare_exchange_weak(previous,end,Ordering::Relaxed,Ordering::Relaxed) {
            Ok(_)=>return Ok(seed),Err(actual)=>previous=actual,
        }
    }
}

#[cfg(test)]mod tests {
    use super::*;
    #[test]fn reserves_whole_parallel_ranges_even_when_clock_stalls_or_goes_backward(){
        let last=AtomicU64::new(0);
        assert_eq!(reserve_at(&last,1000,8).unwrap(),1000);
        assert_eq!(reserve_at(&last,1000,16).unwrap(),1008);
        assert_eq!(reserve_at(&last,999,1).unwrap(),1024);
        assert_eq!(reserve_at(&last,2000,2).unwrap(),2000);
    }
    #[test]fn concurrent_requests_never_overlap_seed_ranges(){
        let last=std::sync::Arc::new(AtomicU64::new(0));
        let threads=(0..16).map(|_|{let last=last.clone();std::thread::spawn(move||{
            (0..32).map(|_|reserve_at(&last,50,8).unwrap()).collect::<Vec<_>>()
        })}).collect::<Vec<_>>();
        let mut seeds=threads.into_iter().flat_map(|thread|thread.join().unwrap())
            .flat_map(|base|(0..8).map(move|offset|base+offset)).collect::<Vec<_>>();
        seeds.sort_unstable();seeds.dedup();assert_eq!(seeds.len(),16*32*8);
    }
    #[test]fn invalid_ranges_fail_without_advancing_the_clock(){
        let last=AtomicU64::new(9);assert!(reserve_at(&last,100,0).is_err());assert_eq!(last.load(Ordering::Relaxed),9);
        assert!(reserve_at(&last,u64::MAX,2).is_err());assert_eq!(last.load(Ordering::Relaxed),9);
        let last=AtomicU64::new(u64::MAX);assert!(reserve_at(&last,0,1).is_err());
    }
    #[test]fn preview_never_reserves_or_changes_the_live_seed_range(){
        let last=AtomicU64::new(8);
        assert_eq!(preview_at(&last,10),10);assert_eq!(preview_at(&last,2),9);
        assert_eq!(last.load(Ordering::Relaxed),8);
        assert!(clock_seed()<(1u64<<53));
    }
}
