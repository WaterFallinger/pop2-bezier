//! Backend-neutral budget decision for a sole FP32 KV cache.
//!
//! `available_bytes` is a best-effort observation made after model weights have
//! been loaded. This module neither queries a GPU nor reserves memory, and it
//! cannot prevent another process from consuming the observed free space later.
//! A false decision retains the compact cache; it never reduces the requested
//! logical lanes or disables CFG. FP32 replaces compact KV, rather than being a
//! second long-lived copy of that KV.

pub const MIN_HEADROOM_BYTES: u64 = 1 << 30;

#[derive(Clone, Copy, Debug)]
pub struct CacheBudgetInput {
    /// Available device memory observed after loading weights; None is unknown.
    pub available_bytes: Option<u64>,
    pub layers: usize,
    pub width: usize,
    /// The model's full maximum context, never the current short prefix length.
    pub max_context: usize,
    pub logical_lanes: usize,
    pub cfg_enabled: bool,
    /// Additional safety/workspace margin. Values below 1 GiB are raised to it.
    pub headroom_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePlanReason {
    Int8History,
    Fp32Fits,
    UnknownAvailableMemory,
    InvalidInput,
    ArithmeticOverflow,
    InsufficientAvailableMemory,
    WorkloadNotSelected,
}

impl CachePlanReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int8History => "INT8 committed history with FP32 token/head scales; floating scratch; expanded FP32 cache disabled",
            Self::WorkloadNotSelected=>"short or small workload keeps compact KV; expanded cache is reserved for measured long multi-lane workloads",
            Self::Fp32Fits => {
                "full-context FP32 KV, templates and headroom fit the observed budget"
            }
            Self::UnknownAvailableMemory => "available memory is unknown; retain compact KV",
            Self::InvalidInput => {
                "invalid cache dimensions or logical lane count; retain compact KV"
            }
            Self::ArithmeticOverflow => "cache budget arithmetic overflowed; retain compact KV",
            Self::InsufficientAvailableMemory => {
                "full-context FP32 KV exceeds the observed budget; retain compact KV"
            }
        }
    }
}

impl std::fmt::Display for CachePlanReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePlan {
    pub use_fp32: bool,
    /// None means the requested count itself could not be represented safely.
    pub physical_rows: Option<u64>,
    /// Maximum-capacity main KV, with two tensors and four bytes per element.
    pub fp32_kv_bytes: Option<u64>,
    /// At most two extra prefill templates for multi-lane preparation. Each
    /// template contains one row, or a conditional/unconditional pair for CFG.
    pub extra_template_bytes: Option<u64>,
    pub headroom_bytes: u64,
    pub required_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub reason: CachePlanReason,
}

pub fn plan_cache(input: CacheBudgetInput) -> CachePlan {
    let headroom = input
        .headroom_bytes
        .unwrap_or(MIN_HEADROOM_BYTES)
        .max(MIN_HEADROOM_BYTES);
    let mut plan = CachePlan {
        use_fp32: false,
        physical_rows: None,
        fp32_kv_bytes: None,
        extra_template_bytes: None,
        headroom_bytes: headroom,
        required_bytes: None,
        available_bytes: input.available_bytes,
        reason: CachePlanReason::InvalidInput,
    };
    if input.layers == 0 || input.width == 0 || input.max_context == 0 || input.logical_lanes == 0 {
        return plan;
    }

    let multiplier = if input.cfg_enabled { 2 } else { 1 };
    plan.physical_rows = u64::try_from(input.logical_lanes)
        .ok()
        .and_then(|n| n.checked_mul(multiplier));
    let bytes_per_row = u64::try_from(input.layers)
        .ok()
        .zip(u64::try_from(input.width).ok())
        .zip(u64::try_from(input.max_context).ok())
        .and_then(|((layers, width), context)| {
            layers
                .checked_mul(width)?
                .checked_mul(context)?
                .checked_mul(2)? // K and V
                .checked_mul(4) // FP32; no additional compact copy
        });
    plan.fp32_kv_bytes = bytes_per_row
        .zip(plan.physical_rows)
        .and_then(|(bytes, rows)| bytes.checked_mul(rows));
    let template_rows = if input.logical_lanes > 1 {
        2 * multiplier
    } else {
        0
    };
    plan.extra_template_bytes = bytes_per_row.and_then(|bytes| bytes.checked_mul(template_rows));
    plan.required_bytes = plan
        .fp32_kv_bytes
        .zip(plan.extra_template_bytes)
        .and_then(|(kv, templates)| kv.checked_add(templates)?.checked_add(headroom));

    let Some(required) = plan.required_bytes else {
        plan.reason = CachePlanReason::ArithmeticOverflow;
        return plan;
    };
    match input.available_bytes {
        None => plan.reason = CachePlanReason::UnknownAvailableMemory,
        Some(available) if available < required => {
            plan.reason = CachePlanReason::InsufficientAvailableMemory
        }
        Some(_) => {
            plan.use_fp32 = true;
            plan.reason = CachePlanReason::Fp32Fits;
        }
    }
    plan
}

/// Conservative performance gate for this model family, separate from the
/// memory guarantee. The measured expansion benefit was in long, multi-lane
/// requests; short/single-lane requests must not blindly pay twice the KV.
pub fn plan_cache_for_workload(input: CacheBudgetInput, prefix_records: usize) -> CachePlan {
    let mut plan = plan_cache(input);
    if plan.use_fp32 && (input.logical_lanes < 4 || prefix_records < 512) {
        plan.use_fp32 = false;
        plan.reason = CachePlanReason::WorkloadNotSelected;
    }
    plan
}

pub fn planned_context_records(prefix: usize, new_primitives: usize, max_context: usize) -> usize {
    prefix
        .saturating_add(new_primitives.min(max_context))
        .min(max_context)
}

#[cfg(test)]
mod workload_tests {
    use super::*;
    #[test]
    fn workload_forecast_is_bounded_and_never_overflows() {
        assert_eq!(planned_context_records(1, 128, 1024), 129);
        assert_eq!(planned_context_records(400, 128, 1024), 528);
        assert_eq!(planned_context_records(usize::MAX, usize::MAX, 1024), 1024);
        assert_eq!(planned_context_records(0, 0, 1024), 0);
    }
    #[test]
    fn expanded_cache_requires_both_measured_workload_and_memory_budget() {
        let input = CacheBudgetInput {
            available_bytes: Some(24 << 30),
            layers: 35,
            width: 1536,
            max_context: 1024,
            logical_lanes: 4,
            cfg_enabled: true,
            headroom_bytes: Some(2 << 30),
        };
        assert!(plan_cache_for_workload(input, 512).use_fp32);
        assert_eq!(
            plan_cache_for_workload(input, 511).reason,
            CachePlanReason::WorkloadNotSelected
        );
        assert!(
            !plan_cache_for_workload(
                CacheBudgetInput {
                    logical_lanes: 1,
                    ..input
                },
                896
            )
            .use_fp32
        );
        assert_eq!(
            plan_cache_for_workload(
                CacheBudgetInput {
                    available_bytes: Some(1),
                    ..input
                },
                896
            )
            .reason,
            CachePlanReason::InsufficientAvailableMemory
        );
        assert_eq!(
            plan_cache_for_workload(
                CacheBudgetInput {
                    available_bytes: None,
                    ..input
                },
                896
            )
            .reason,
            CachePlanReason::UnknownAvailableMemory
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const ROW: u64 = 35 * 1536 * 1024 * 2 * 4;

    fn input(lanes: usize, cfg: bool) -> CacheBudgetInput {
        CacheBudgetInput {
            available_bytes: Some(24 * GIB),
            layers: 35,
            width: 1536,
            max_context: 1024,
            logical_lanes: lanes,
            cfg_enabled: cfg,
            headroom_bytes: None,
        }
    }

    #[test]
    fn exact_main_and_template_bytes_count_physical_cfg_rows() {
        let one = plan_cache(input(1, false));
        assert_eq!(ROW, 440_401_920);
        assert_eq!(one.physical_rows, Some(1));
        assert_eq!(one.fp32_kv_bytes, Some(ROW));
        assert_eq!(one.extra_template_bytes, Some(0));
        assert_eq!(one.required_bytes, Some(ROW + GIB));
        assert!(one.use_fp32);

        let guided_one = plan_cache(input(1, true));
        assert_eq!(guided_one.physical_rows, Some(2));
        assert_eq!(guided_one.fp32_kv_bytes, Some(2 * ROW));
        assert_eq!(guided_one.extra_template_bytes, Some(0));

        let four = plan_cache(input(4, false));
        assert_eq!(four.physical_rows, Some(4));
        assert_eq!(four.fp32_kv_bytes, Some(4 * ROW));
        assert_eq!(four.extra_template_bytes, Some(2 * ROW));
        assert_eq!(four.required_bytes, Some(6 * ROW + GIB));

        let guided_four = plan_cache(input(4, true));
        assert_eq!(guided_four.physical_rows, Some(8));
        assert_eq!(guided_four.fp32_kv_bytes, Some(8 * ROW));
        assert_eq!(guided_four.extra_template_bytes, Some(4 * ROW));
        assert_eq!(guided_four.required_bytes, Some(12 * ROW + GIB));
    }

    #[test]
    fn exact_budget_boundary_and_one_byte_short_do_not_change_lanes() {
        let mut parameters = input(8, true);
        let estimate = plan_cache(parameters);
        let required = estimate.required_bytes.unwrap();
        parameters.available_bytes = Some(required);
        let fits = plan_cache(parameters);
        assert!(fits.use_fp32);
        assert_eq!(fits.reason, CachePlanReason::Fp32Fits);
        parameters.available_bytes = Some(required - 1);
        let short = plan_cache(parameters);
        assert!(!short.use_fp32);
        assert_eq!(short.reason, CachePlanReason::InsufficientAvailableMemory);
        assert_eq!(short.physical_rows, Some(16));
        assert_eq!(short.fp32_kv_bytes, fits.fp32_kv_bytes);
        assert_eq!(short.required_bytes, fits.required_bytes);
    }

    #[test]
    fn unknown_or_zero_free_memory_is_conservative_but_keeps_estimates() {
        let mut parameters = input(4, true);
        parameters.available_bytes = None;
        let unknown = plan_cache(parameters);
        assert!(!unknown.use_fp32);
        assert_eq!(unknown.reason, CachePlanReason::UnknownAvailableMemory);
        assert_eq!(unknown.physical_rows, Some(8));
        assert_eq!(unknown.fp32_kv_bytes, Some(8 * ROW));
        assert_eq!(unknown.required_bytes, Some(12 * ROW + GIB));
        parameters.available_bytes = Some(0);
        assert_eq!(
            plan_cache(parameters).reason,
            CachePlanReason::InsufficientAvailableMemory
        );
    }

    #[test]
    fn headroom_has_one_gib_minimum_and_respects_larger_requested_margin() {
        let mut parameters = input(1, false);
        parameters.headroom_bytes = Some(0);
        assert_eq!(plan_cache(parameters).headroom_bytes, GIB);
        parameters.headroom_bytes = Some(3 * GIB);
        let plan = plan_cache(parameters);
        assert_eq!(plan.headroom_bytes, 3 * GIB);
        assert_eq!(plan.required_bytes, Some(ROW + 3 * GIB));
    }

    #[test]
    fn zero_dimensions_or_lanes_never_select_fp32() {
        for field in 0..4 {
            let mut parameters = input(4, true);
            match field {
                0 => parameters.layers = 0,
                1 => parameters.width = 0,
                2 => parameters.max_context = 0,
                _ => parameters.logical_lanes = 0,
            }
            let plan = plan_cache(parameters);
            assert!(!plan.use_fp32);
            assert_eq!(plan.reason, CachePlanReason::InvalidInput);
            assert_eq!(plan.required_bytes, None);
        }
    }

    #[test]
    fn multiplication_and_sum_overflow_fail_closed() {
        let mut parameters = input(4, true);
        parameters.layers = usize::MAX;
        parameters.width = usize::MAX;
        let overflow = plan_cache(parameters);
        assert!(!overflow.use_fp32);
        assert_eq!(overflow.reason, CachePlanReason::ArithmeticOverflow);
        assert_eq!(overflow.physical_rows, Some(8));
        assert_eq!(overflow.fp32_kv_bytes, None);

        let mut parameters = input(1, false);
        parameters.available_bytes = Some(u64::MAX);
        parameters.headroom_bytes = Some(u64::MAX);
        let overflow = plan_cache(parameters);
        assert!(!overflow.use_fp32);
        assert_eq!(overflow.reason, CachePlanReason::ArithmeticOverflow);
        assert_eq!(overflow.fp32_kv_bytes, Some(ROW));
        assert_eq!(overflow.extra_template_bytes, Some(0));
        assert_eq!(overflow.required_bytes, None);

        if usize::BITS == 64 {
            let mut parameters = input(usize::MAX, true);
            parameters.available_bytes = Some(u64::MAX);
            let overflow = plan_cache(parameters);
            assert!(!overflow.use_fp32);
            assert_eq!(overflow.reason, CachePlanReason::ArithmeticOverflow);
            assert_eq!(overflow.physical_rows, None);
        }
    }
}
