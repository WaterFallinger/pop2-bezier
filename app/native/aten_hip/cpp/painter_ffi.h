#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PP_ATEN_EXPORTS)
#define PP_ATEN_API __declspec(dllexport)
#elif defined(_WIN32)
#define PP_ATEN_API __declspec(dllimport)
#else
#define PP_ATEN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PpAtenModel PpAtenModel;
typedef struct PpAtenCache PpAtenCache;

// ABI version 1. Paths and returned error/info strings are UTF-8.
// 0=success, 1=precondition/preparation failure (no user-cache execution),
// 2=invalid model output (failed frame not delivered),
// 3=execution uncertain/device failure (never replay; reload required).
// CAPTURE_STATE_UNCERTAIN specifically requires application restart: all
// further load/prepare/configure/prefill/frame calls in this process are refused.
typedef struct PpAtenLoadOptions {
    uint32_t struct_size;
    uint32_t device_ordinal;
    uint32_t prefer_efficient_attention;
    uint32_t reserved;
    const char* index_utf8; // Native Rust metadata index; tensor bytes stay in original model.pt.
} PpAtenLoadOptions;

typedef struct PpAtenFrameOptions {
    uint32_t struct_size;
    uint32_t logical_rows;
    uint32_t guidance;       // 1 or 2; physical rows are conditional/unconditional interleaved.
    uint32_t graph_allowed;
    uint32_t sampling_gpu;   // false => explicit CPU f64 sampling, no graph.
    uint32_t efficient_attention;
    double cfg;
    double alpha;
} PpAtenFrameOptions;

typedef struct PpAtenAutoProbeOptions {
    uint32_t struct_size;
    uint32_t logical_rows;
    uint32_t guidance;
    uint32_t regional;
    uint32_t candidate_efficient;
    uint32_t candidate_gpu_sampling;
    uint32_t prefix_records;
    uint32_t reserved;
    double cfg;
    double alpha;
} PpAtenAutoProbeOptions;

PP_ATEN_API uint32_t pp_aten_abi_version(void);
PP_ATEN_API const char* pp_aten_last_error(void);
PP_ATEN_API int32_t pp_aten_load(const char* model_utf8, const PpAtenLoadOptions* options,
                               PpAtenModel** out_model);
PP_ATEN_API const char* pp_aten_model_info(const PpAtenModel* model);
PP_ATEN_API int32_t pp_aten_configure(PpAtenModel* model, uint32_t efficient_prefill);
PP_ATEN_API int32_t pp_aten_begin_request(PpAtenModel* model);
// Additive ABI1 extension: 0=Auto,1=Enabled,2=Disabled. Missing on older DLLs.
PP_ATEN_API int32_t pp_aten_configure_kv(PpAtenModel* model, uint32_t preference);
PP_ATEN_API int32_t pp_aten_prepare(PpAtenModel* model);
PP_ATEN_API int32_t pp_aten_prepare_graph(PpAtenModel* model);
PP_ATEN_API int32_t pp_aten_probe_auto(PpAtenModel* model, const PpAtenAutoProbeOptions* options);
PP_ATEN_API const char* pp_aten_probe_info(const PpAtenModel* model);
PP_ATEN_API void pp_aten_model_release(PpAtenModel* model);

// All inputs are copied/consumed during this call. records[N,9], text[W].
// text_mask is optional U8[W]; offset supports compact unconditional BOS/EOS.
// Caller provides out_terms[2]. The cache retains immutable pending-X logits.
PP_ATEN_API int32_t pp_aten_prefill(PpAtenModel* model, const uint32_t* records,
                                  size_t record_count, const uint32_t* text,
                                  size_t text_count, const uint8_t* text_mask,
                                  size_t position_offset, PpAtenCache** out_cache,
                                  float* out_terms);
// Additive ABI1 extension: immutable prefix reuse + visual suffix prefill.
// retained_tokens counts physical cached positions, including text/BOS.
PP_ATEN_API int32_t pp_aten_extend(PpAtenModel* model,const PpAtenCache* source,
                                 size_t retained_tokens,const uint32_t* suffix,
                                 size_t suffix_records,PpAtenCache** out_cache,float* out_terms);
PP_ATEN_API int32_t pp_aten_cache_clone(const PpAtenCache* cache, PpAtenCache** out_cache);
PP_ATEN_API int32_t pp_aten_cache_info(const PpAtenCache* cache,
                                     size_t* out_length, size_t* out_absolute_position);
PP_ATEN_API void pp_aten_cache_release(PpAtenCache* cache);

// bounds: U32[9,logical_rows,2] relative (start,count) within each head.
// uniforms: F32[9,logical_rows], lane-local randomness supplied once/frame.
// tokens: caller U32[logical_rows,9], terms: F32[physical_rows,2],
// valid: caller U32[logical_rows], out_caches: physical_rows owned new handles.
// Input snapshots remain usable for a branch; pending logits are never views
// into a graph buffer overwritten by the next frame. Release each new handle.
PP_ATEN_API int32_t pp_aten_frame(PpAtenModel* model,
                                const PpAtenCache* const* caches,
                                const PpAtenFrameOptions* options,
                                const uint32_t* bounds, const float* uniforms,
                                uint32_t* out_tokens, float* out_terms,
                                uint32_t* out_valid, PpAtenCache** out_caches);

// Per-model diagnostics only, JSON remains valid until the next operation.
PP_ATEN_API const char* pp_aten_diagnostics(const PpAtenModel* model);

#ifdef __cplusplus
}

#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace pp {
// The ABI stores logical rows as U32; host buffers use size_t and tensors I64.
// These are representation checks, not a product-specific parallelism limit.
struct RowGeometry {
  size_t logical,physical,attributes,bounds,terms;
};
inline size_t checked_row_product(size_t count,size_t factor) {
  if(factor&&count>std::numeric_limits<size_t>::max()/factor)
    throw std::overflow_error("native batch dimensions exceed address-space representation");
  const auto value=count*factor;
  if(value>uint64_t(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("native batch dimensions exceed tensor index representation");
  return value;
}
inline RowGeometry row_geometry(uint32_t logical,uint32_t guidance) {
  if(!logical||(guidance!=1&&guidance!=2))
    throw std::invalid_argument("native parallel rows must be positive and guidance must be1 or2");
  RowGeometry result{logical,checked_row_product(logical,guidance),
    checked_row_product(logical,9),checked_row_product(logical,18),
    checked_row_product(checked_row_product(logical,guidance),2)};
  for(const auto count:{result.bounds,result.attributes,result.terms,result.physical})
    if(checked_row_product(count,sizeof(uint64_t))>size_t(std::numeric_limits<ptrdiff_t>::max()))
      throw std::overflow_error("native batch buffers exceed this process address space");
  return result;
}
}
#endif
