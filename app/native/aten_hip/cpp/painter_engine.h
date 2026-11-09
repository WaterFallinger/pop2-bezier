#pragma once
#include "painter_model.h"
#include "painter_ffi.h"
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <memory>
#include <mutex>

namespace pp {
struct RowLease {};
struct CacheOwner {
  Cache cache;
  uint64_t model_id=0, id=0;
  std::vector<int64_t> current_lengths, current_absolute;
  std::vector<std::vector<std::weak_ptr<RowLease>>> row_leases;
  bool poisoned=false;
};
struct PendingSnapshot { at::Tensor attributes,termination; };
struct CacheSnapshot {
  std::shared_ptr<CacheOwner> owner;
  std::shared_ptr<PendingSnapshot> pending;
  int64_t row=0, pending_row=0, length=0, absolute=0;
  std::shared_ptr<RowLease> lease;
};
struct FrameResult {
  std::vector<uint32_t> tokens,valid;
  std::vector<float> terms;
  std::vector<CacheSnapshot> caches;
};
struct SamplingResult { at::Tensor tokens,valid,support; };
SamplingResult sample_gpu(const at::Tensor& raw,const at::Tensor& uniforms,
    const at::Tensor& bounds,int64_t guidance,float cfg,float alpha,int64_t offset);

struct DeviceSelection {
  int ordinal=0,count=0;
  bool automatic=false;
  std::string candidates_json="[]", reason;
};

class Engine {
 public:
  Engine(const std::string& model_utf8,const std::string& index_utf8,
      uint32_t ordinal,bool prefer_efficient);
  ~Engine();
  CacheSnapshot prefill(const Prefix& prefix,float* out_terms);
  CacheSnapshot extend(const CacheSnapshot& source,int64_t retained,const std::vector<std::array<int64_t,9>>& suffix,float* out_terms);
  void configure(bool efficient_prefill);
  void begin_request();
  void configure_kv(uint32_t preference);
  void prepare_optional();
  void prepare_graph();
  void probe_auto(const PpAtenAutoProbeOptions& options);
  const std::string& probe_info() const noexcept { return probe_info_; }
  FrameResult frame(const std::vector<CacheSnapshot>& caches,const PpAtenFrameOptions& options,
      const uint32_t* bounds,const float* uniforms);
  const std::string& info() const noexcept { return info_; }
  const std::string& diagnostics() const noexcept { return diagnostics_; }
  uint64_t id() const noexcept { return id_; }
 private:
  struct Session;
  DeviceSelection selection_;
  int ordinal_=0,device_count_=0;
  uint64_t id_=0;
  c10::hip::HIPStreamMasqueradingAsCUDA stream_;
  std::shared_ptr<Model> model_;
  std::shared_ptr<Session> session_;
  bool efficient_available_=false,sampling_available_=false,graph_available_=false,poisoned_=false;
  bool efficient_prefill_=false;
  bool optional_preparation_attempted_=false;
  bool graph_preparation_attempted_=false;
  uint32_t kv_preference_=2;
  bool kv_preparation_attempted_=false,kv_available_=false;
  std::string kv_reason_="INT8 KV not requested; BF16 retained";
  std::string kv_unavailable_reason_;
  uint64_t graph_captures_=0,graph_replays_=0,frame_calls_=0,ordinary_forwards_=0;
  uint64_t auto_probe_calls_=0;
  uint64_t full_prefill_calls_=0,extended_prefill_calls_=0,reused_prefix_tokens_=0;
  uint64_t session_allocations_=0,session_retargets_=0,session_retarget_rows_=0;
  std::string last_graph_error_,info_,diagnostics_,probe_info_="{}";
  std::string efficient_disabled_,sampling_disabled_,graph_disabled_;
  std::mutex mutex_;
  void update_diagnostics();
  void update_info();
  void verify_capabilities(bool prepare_optional);
  std::string kv_info()const;
};
std::string json_quote(const std::string& value);
bool execution_uncertain(const std::string& message);
}
